import os
import json
import tempfile
import httpx
from fastapi import FastAPI
from pymongo import MongoClient
from fastapi.responses import JSONResponse
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from datetime import datetime
import pytz
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId


# Load environment variables
load_dotenv()

# setting our ocr and llm API keys and endpoints
form_recognizer_endpoint = os.getenv("AZURE_OCR_ENDPOINT")
form_recognizer_key = os.getenv("AZURE_OCR_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")
database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")

document_analysis_client = DocumentAnalysisClient(
    endpoint=form_recognizer_endpoint, credential=AzureKeyCredential(form_recognizer_key)
)

# initialize openAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# initialize fastAPI 
app = FastAPI()

# MongoDB client setup
mongo_client = MongoClient(mongo_uri)
db = mongo_client[database_name]
collection = db[collection_name]

# Functions
def analyze_document(document_path):
    try:
        with open(document_path, "rb") as f:
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-document", document=f.read()
            )
        result = poller.result()

        extracted_data = []
        for page in result.pages:
            page_content = " ".join([line.content for line in page.lines])
            extracted_data.append({str(page.page_number - 1): page_content})
        # print(extracted_data)
        return extracted_data
    except Exception as e:
        print(f"Error during document analysis: {e}")
        raise

def get_openai_response(messages):
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error fetching response from OpenAI: {e}")
        raise

def process_ocr_output(ocr_output):
    try:
        corrected_output_parts = []
        
        for page in ocr_output:
            page_content = list(page.values())[0]
            messages = [
                {"role": "system", "content": "You are a helpful assistant that fixes errors in OCR outputs and provides correct data in the same JSON format."},
                {"role": "user", "content": f"Fix the errors and get correct data in same JSON format:\n{page_content}"}
            ]
            response = get_openai_response(messages)
            corrected_output_parts.append({list(page.keys())[0]: response})
        
        return corrected_output_parts
    except json.JSONDecodeError as e:
        print(f"Error parsing corrected JSON: {e}")
        raise

def process_document(doc):
    try:
        collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "processing"}})

        # Check if OCR output already exists
        ocr_output = doc.get("ocr_output")
        if ocr_output:
            extracted_data = ocr_output
            processed_data = process_ocr_output(extracted_data)
            collection.update_one({"_id": doc["_id"]}, {"$set": {"ocr_output": extracted_data, "json_data": processed_data}})
        else:
            temp_file_path = os.path.join(tempfile.gettempdir(), secure_filename(os.path.basename(doc["image"])))

            response = httpx.get(doc["image"])
            response.raise_for_status()

            with open(temp_file_path, 'wb') as file:
                file.write(response.content)

            try:
                extracted_data = analyze_document(temp_file_path)
                if not extracted_data:
                    collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed"}})
                    print(f"No data extracted for document {doc['_id']}")
                    return
                else:
                    processed_data = process_ocr_output(extracted_data)
                    collection.update_one({"_id": doc["_id"]}, {"$set": {"ocr_output": extracted_data, "json_data": processed_data}})
            except Exception as e:
                print(f"Document analysis(OCR) failed for the document {doc['_id']}: {e}")
                collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed"}})
                return
            finally:
                os.remove(temp_file_path)

        processed_date = datetime.now(pytz.timezone('UTC')).isoformat()
        object_id = doc["_id"]
        try:
            final_assignment = get_metadata_for_final_assignment(object_id, collection)
        except Exception as e:
            print(f"Error getting final_assignment for document {object_id}: {e}")
            
            try:
                final_assignment = get_metadata_for_final_assignment(object_id, collection)
            except Exception as e:
                print(f"Retry failed for final_assignment for document {object_id}: {e}")
    
        try:
            final_release = get_metadata_for_final_release(object_id, collection)
        except Exception as e:
            print(f"Error getting final_release for document {object_id}: {e}")
            try:
                final_release = get_metadata_for_final_release(object_id, collection)
            except Exception as e:
                print(f"Retry failed for final_release for document {object_id}: {e}")
        update_data = {
            "status": "processed",
            "processed_date": processed_date,
            "final_release": final_release,
            "final_assignment": final_assignment
        }

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": update_data}
        )
    except Exception as e:
        print(f"Processing failed for document {doc['_id']}: {e}")
        collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "failed"}})
def process_documents():
    documents = collection.find({"status": {"$in": ["notprocessed", "processing", "failed"]}})
    for doc in documents:
        process_document(doc)

# Function call
@app.get("/")
def read_root():
    return {"status": "success"}


@app.post("/process")
def process_route():
    process_documents()
    return JSONResponse(content={"message": "Documents processed successfully"}, status_code=200)

def get_metadata(object_id, collection, fields):
    document = collection.find_one({"_id": ObjectId(object_id)})
    if not document:
        raise ValueError(f"Document not found: {object_id}")

    json_data = document.get("json_data", [])
    combined_content = " ".join([list(page.values())[0] for page in json_data])

    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts specific information from document content."},
        {"role": "user", "content": f"Extract the following information from this content, filling in the values for each field:\n{combined_content}\n\nFields: {json.dumps(fields, indent=2)}"}
    ]

    

    try:
        response = get_openai_response(messages)
        return json.loads(response)
    except json.JSONDecodeError as e:

        print(f"Failed to parse OpenAI response for metadata : {e}")

def get_metadata_for_final_assignment(object_id, collection):
    fields = {"Record Type 'Z'" : "",
                "Document Type (Must be populated with one of the valid codes)": "",
                "FIPS Code": "",
                "MERS Indicator (ASSIGNEE)" : "",
                "RECORD ID M = MAIN Record (default value for all non-addendum records) A = APN Addendum; D= DOT Addendum" : "",
                "Assignment Recording Date" : "",
                "Assignment EFFECTIVE or CONTRACT Date." : "",
                "Assignment Document Number (also known as:  Reception No, Instrument No.)" : "",
                "Assignment Book Number" : "",
                "Assignment Page Number" : "",
                "Multiple Page Image Flag" : "",
                "LPS Image Identifier" : "",
                "Original Deed of Trust ('DOT') Recording Date" : "",
                "Original Deed of Trust ('DOT') Contract Date" : "",
                "Original Deed of Trust('DOT') Document Number" : "", 
                "Original Deed of Trust ('DOT') Book Number" : "",
                "Original Deed of Trust ('DOT') Page Number" : "",
                "Original Beneficiary/Lender/Mortgagee/In Favor of/Made By": "",
                "Original Loan Amount" : "",
                "Assignor Name(s) " : "",
                "Loan Number" : "",
                "Assignee(s) (Lender(s) receiving right, title & interest in the Deed of Trust or Mortgage)" : "",
                "MERS (MIN) Number" : "",
                "MERS NUMBER  PASS VALIDATION" : "",
                "Assignee / Pool" : "",
                "MSP Servicer Number and Loan Number" : "",
                "Borrower Name(s)/Corporation(s)" : "",
                "Assessor Parcel Number (APN, PIN, PID)" : "",
                "Multiple APN Code" : "",
                "Tax Acct ID" : "",
                "Property: Full Street Address-  (Look for phrases such as `Commonly known as`)" : "",
                "Property: Unit #" : "",
                "Property: City Name" : "",
                "Property: State" : "",
                "Property: Zip" : "",
                "Property: Zip + 4" : "",
                "Data Entry Date": "",
                "Data Entry Operator Code" : "",
                "Vendor Source Code" : ""
                }
    return get_metadata(object_id, collection, fields)

def get_metadata_for_final_release(object_id, collection):
    fields = {
        "Record Type" : "",
        "Document Type (Must be populated with one of the valid codes)" : "",
        "FIPS Code" : "",
        "RECORD ID M = MAIN Record (default value for all non-addendum records); A = APN Addendum; D= DOT Addendum" : "",
        "Release Recording Date" : "",
        "Release Contract Date or Effective Date " : "",
        "(Mortgage) Payoff Date (P.O. Date)" : "",
        "Release Document Number (Instrument, Reception No)" : "",
        "Release Book Number (Folio, Liber,Volume)" : "",
        "Release Page Number" : "",
        "Multiple Page Image Flag" : "",
        "LPS Image Identifier" : "",
        "Original Deed of Trust ('DOT') Recording Date" : "",
        "Original Deed of Trust ('DOT') Contract Date" : "",
        "Original Deed of Trust('DOT') Document Number" : "",
        "Original Deed of Trust ('DOT') Book Number" : "",
        "Original Deed of Trust ('DOT') Page Number" : "",
        "Original Beneficiary/Lender/Mortgagee" : "",
        "Original Loan Amount" : "",
        "Loan Number" : "",
        "Current Beneficiary/Lender/Mortgagee" : "",
        "MERS (MIN) Number" : "",
        "MERS NUMBER  PASS VALIDATION" : "",
        "MSP Servicer Number and Loan Number" : "",
        "Current Lender 'Pool'": "",
        "Borrower Name(s)/Corporation(s)" : "",
        "Borrower Mail Full Street Address " : "",
        "Borrower Mail Unit" : "",
        "Borrower Mail City Name" : "",
        "Borrower Mail State" : "",
        "Borrower Mail Zip" : "",
        "Borrower Mail Zip + 4" : "",
        "Assessor Parcel Number (APN, PID, PIN)" : "",
        "Multiple APN Code" : "",
        "Tax Acct ID" : "",
        "Property: Full Street Address-  (Look for phrases such as `Commonly known as`)" : "",
        "Property: Unit #" : "",
        "Property: City Name" : "",
        "Property: State" : "",
        "Property: Zip" : "",
        "Property: Zip + 4" : "",
        "Data Entry Date" : "",
        "Data Entry Operator Code" : "",
        "Vendor Source Code" : ""
                    }
    return get_metadata(object_id, collection, fields)

# aps schedular
scheduler = BackgroundScheduler()
scheduler.add_job(process_documents, 'interval', seconds=5)
scheduler.start()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
