import streamlit as st
import lang_chain_helper

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a cuisine:", ("Indian", "Italian", "Arabic", "Chinese","Arabic","Mexican"))



if cuisine:
    response = lang_chain_helper.generate_restaurant_name(cuisine)
    st.header(response['restaurant_name'].strip())          # fixed line
    menu_item = response['menu_item'].strip().split(",")
    st.write("**Menu-item**")
    for item in menu_item:
        st.write("~", item)
