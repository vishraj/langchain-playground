import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google")

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Quick sanity check to verify Python actually sees your variables:
print("Tracing enabled:", os.environ.get("LANGSMITH_TRACING"))
print("API Key loaded:", bool(os.environ.get("LANGSMITH_API_KEY")))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def main():
    print("Hello from langchain!")
    information = """
      Satya Narayana Nadella was born on 19 August 1967 in a Telugu Hindu family in Hyderabad. His father, Bukkapuram Nadella Yugandhar, was an Indian Administrative Service officer of the 1962 batch. His mother Prabhavati was a Sanskrit lecturer.[16] Yugandhar hailed from Bukkapuram in Anantapur district of Andhra Pradesh;[17][18] his own father had migrated to Bukkapuram from Nadella village in Guntur district (present-day Palnadu district) of Andhra Pradesh.[19][20]

      Nadella attended the Hyderabad Public School, Begumpet before receiving a bachelor's degree in electrical engineering from the Manipal Institute of Technology in Karnataka in 1988. He then traveled to the United States to study for an MS in computer science at the University of Wisconsin–Milwaukee, receiving his degree in 1990. He received an MBA from the University of Chicago Booth School of Business in 1997.  
    """

    summary_template = """
        given the information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts abouyt them
    """    

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        verbose=True,
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    chain = summary_prompt_template | llm | StrOutputParser()
    response = chain.invoke({"information": information})
    print(response)
    
if __name__ == "__main__":
    main()
