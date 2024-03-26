import streamlit as st
import pdfplumber
from PIL import Image

page_icon = Image.open("./favicon.ico")
st.set_page_config(page_title="Smart Fill", page_icon=page_icon, layout="wide", initial_sidebar_state="expanded")
logo = Image.open("./favicon.ico")
    


        

# Function to extract attributes from PDF document using pdfplumber
def extract_attributes_from_pdf(file_path):
    attributes = {}

    with pdfplumber.open(file_path) as pdf:
        first_page = pdf.pages[0]  # Assuming attributes are on the first page

        # Extract text from the first page
        text = first_page.extract_text()

        attribute_pairs = [
            ("Property",),
            ("Borrower(s)", "Borrower"),
            ("Seller(s)", "Seller"),
            ("Loan Amount",),
            ("Loan Term",),
            ("Lender",),
            ("Cash to Close",)
        ]

        for pair in attribute_pairs:
            for attribute in pair:
                if attribute in text:
                    attribute_index = text.index(attribute)
                    if attribute != pair[0]:
                        end_index = text.find("\n", attribute_index)
                    else:
                        end_index = text[attribute_index:].find("\n") + attribute_index
                    attribute_value = text[attribute_index + len(attribute):end_index].strip()
                    attributes[attribute] = attribute_value  # Store using the exact attribute name as in the PDF
                    break

    return attributes

# Streamlit UI

st.sidebar.image("FamiologyTextLogo.png", use_column_width=True)

st.markdown(

    """

    <style>

        section[data-testid="stSidebar"] {

            width: 800px !important; # Set the width to your desired value

        }

    </style>

    """,

    unsafe_allow_html=True,

)

        
def main():
    st.title("Smart Fill - Famiology.io")


    
     # Section 1: Information about Smart Fill
    with st.sidebar.container():
        with st.expander("Smart Fill: Automatic Data Prepopulation"):
            st.write("This intelligent feature automatically populates relevant data fields based on the document you upload. Say goodbye to manual data entry and let Smart Fill do the heavy lifting for you.")
        
    # Section 2: Usage Instructions
    with st.sidebar.container():
        with st.expander("How It Works"):
            st.write("1. Upload Your Document: Simply upload your document using the provided file upload button.\n"
                        "2. Smart Analysis: Our system analyzes the document to identify key data points such as names, addresses, dates, and more. Further we can configure it with reference to domain.\n"
                        "3. Automatic Prepopulation: Once the analysis is complete, Smart Fill intelligently fills in the corresponding fields in your form, saving you time and effort.\n"
                        "4. Review and Edit: You always have the final say. Review the pre populated data, make any necessary edits, and proceed with confidence")
        
    # Section 3: Additional Notes
    with st.sidebar.container():
        with st.expander("What problem it solves:"):
            st.write("Client Experience: Busy or unmotivated clients  expect automated pre-filled information sourced from AI or backend operations.\n"
                    "Efficiency: \n" 
                    "Cut down on manual data entry and reduce processing time.")
        
        
    # File upload
    uploaded_file = st.file_uploader("Upload document (PDF)", type="pdf")

    if uploaded_file:
        # Extract attributes from PDF document using pdfplumber
        attributes = extract_attributes_from_pdf(uploaded_file)
        if attributes:
            st.subheader("Pre-filled Attributes")

            for key, value in attributes.items():
                st.text_input(key, value)
        else:
            st.info("No attributes identified.")

if __name__ == "__main__":
    main()
