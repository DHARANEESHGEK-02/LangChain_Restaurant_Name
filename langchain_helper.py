import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.chains import SequentialChain


load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(mistral_api_key=api_key, temperature=0.6)


def generate_restaurant_name(cuisine):

    # 1) Cuisine → restaurant name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],      # fixed name: input_variables
        template=(
            "I want to open a restaurant for a {cuisine} food. "
            "Give only one unique name."
        ),
    )
    cuisine_chain = LLMChain(
        llm=llm,
        prompt=prompt_template_name,
        output_key="restaurant_name",    # not output_keys
    )

    # 2) Restaurant name → menu items
    prompt_template_menu = PromptTemplate(
        input_variables=["restaurant_name"],   # already correct
        template=(
            "Suggest some menu items for {restaurant_name}. "
            "Return as comma-separated values."
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

    response = chain({"cuisine": cuisine})
    return response
if __name__=="__main__":
    print(generate_restaurant_name("Chinese"))
