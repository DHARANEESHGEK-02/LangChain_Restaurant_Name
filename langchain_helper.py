import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, SequentialChain

# -------- API KEY HANDLING --------
# On local: you can keep a .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not required on Streamlit Cloud (uses secrets)
    pass

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise RuntimeError("MISTRAL_API_KEY is not set. Check .env locally or Streamlit secrets.")

# Explicitly set a model name
llm = ChatMistralAI(
    mistral_api_key=api_key,
    model="mistral-small-latest",   # or another valid model from your Mistral account
    temperature=0.6,
)


def generate_restaurant_name(cuisine: str):
    # 1) Cuisine → restaurant name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template=(
            "I want to open a restaurant for a {cuisine} food. "
            "Give only one unique, creative restaurant name."
        ),
    )

    cuisine_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_name,
        output_key="restaurant_name",
    )

    # 2) Restaurant name → menu items
    prompt_template_menu = PromptTemplate(
        input_variables=["restaurant_name"],
        template=(
            "Suggest some menu items for the restaurant called {restaurant_name}. "
            "Return only the items as comma-separated values."
        ),
    )

    resta_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_menu,
        output_key="menu_item",
    )

    chain = SequentialChain(
        chains=[cuisine_chain, resta_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_item"],
        verbose=True,
    )

    return chain({"cuisine": cuisine})


if __name__ == "__main__":
    print(generate_restaurant_name("Chinese"))
