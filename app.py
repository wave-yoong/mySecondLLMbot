import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import json
import glob

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="HAI-5014's Second Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

if "system_message" not in st.session_state:
    st.session_state.system_message = "You are a helpful assistant."

if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = []

if "selected_experiment" not in st.session_state:
    st.session_state.selected_experiment = None

if "selected_condition" not in st.session_state:
    st.session_state.selected_condition = None

def load_experiments():
    """Load all experiment JSON files from the prompts directory"""
    experiments = []
    prompts_dir = os.path.join(os.getcwd(), "prompts")
    
    if not os.path.exists(prompts_dir):
        return experiments
    
    json_files = glob.glob(os.path.join(prompts_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                experiments.append(data)
        except Exception as e:
            st.warning(f"Error loading {file_path}: {str(e)}")
    
    return experiments

def get_openai_client():
    """Create and return an OpenAI client configured with environment variables"""
    token = os.getenv("GITHUB_TOKEN")
    endpoint = os.getenv("GITHUB_ENDPOINT", "https://models.github.ai/inference")
    
    if not token:
        st.error("GitHub token not found in environment variables. Please check your .env file.")
        st.stop()
        
    return OpenAI(
        base_url=endpoint,
        api_key=token,
    )

def generate_response(prompt, system_message):
    """Generate a response from the model and track usage"""
    client = get_openai_client()
    model_name = os.getenv("GITHUB_MODEL", "openai/gpt-4o")
    
    # Prepare messages by including all history and the system message
    messages = [{"role": "system", "content": system_message}]
    
    # Add all previous messages from history
    for msg in st.session_state.messages:
        if msg["role"] != "system":  # Skip system messages as we've already added it
            messages.append(msg)
    
    # Add the new user message
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            stream=True,
            stream_options={'include_usage': True}
        )
        
        # Container for the assistant's response
        response_container = st.chat_message("assistant")
        full_response = ""
        usage = None
        
        # Stream the response
        message_placeholder = response_container.empty()
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                full_response += content_chunk
                message_placeholder.markdown(full_response + "▌")
            if chunk.usage:
                usage = chunk.usage
        
        # Update the final response without the cursor
        message_placeholder.markdown(full_response)
        
        # Add the message to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Store usage stats if available
        if usage:
            usage_dict = usage.dict()
            st.session_state.usage_stats.append({
                "prompt_tokens": usage_dict.get("prompt_tokens", 0),
                "completion_tokens": usage_dict.get("completion_tokens", 0),
                "total_tokens": usage_dict.get("total_tokens", 0)
            })
        
        return True
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return False

# UI Layout
st.title("🤖 GPT-4o Chatbot")

# Sidebar for settings
with st.sidebar:
    st.subheader("Settings")
    
    # System message editor - use the value from session_state directly
    system_message_value = st.session_state.system_message
    st.text_area(
        "Edit System Message", 
        value=system_message_value,
        key="system_message_input",
        height=150
    )
    
    if st.button("Update System Message"):
        st.session_state.system_message = st.session_state.system_message_input
        st.success("System message updated!")
    
    # Experiment loader section
    st.markdown("---")
    st.subheader("Experiment Loader")
    
    experiments = load_experiments()
    if experiments:
        # First dropdown: select experiment with a blank default option
        experiment_names = ["Select an experiment"] + [exp.get('experiment_name', "Unnamed Experiment") for exp in experiments]
        exp_index = st.selectbox("Select Experiment", 
                               range(len(experiment_names)),
                               format_func=lambda i: experiment_names[i]) 
        
        # Only proceed if a valid experiment is selected (not the blank option)
        if exp_index > 0:
            selected_experiment = experiments[exp_index - 1]  # Adjust index for the actual experiment
            
            # Second dropdown: select condition within the experiment with a blank default
            if 'conditions' in selected_experiment and selected_experiment['conditions']:
                condition_names = ["Select a condition"] + [cond.get('label', f"Condition {i+1}") 
                                  for i, cond in enumerate(selected_experiment['conditions'])]
                cond_index = st.selectbox("Select Condition", 
                                        range(len(condition_names)),
                                        format_func=lambda i: condition_names[i])
                
                # Only enable the load button if a valid condition is selected
                if cond_index > 0:
                    selected_condition = selected_experiment['conditions'][cond_index - 1]  # Adjust index
                    
                    # Preview the system message
                    st.markdown("### Preview: System Message")
                    system_prompt = selected_condition.get('system_prompt', "You are a helpful assistant.")
                    st.text_area("", value=system_prompt, height=120, disabled=True, key="preview_system_message")
                    
                    # Load button
                    if st.button("Load Experiment"):
                        # Update system message
                        st.session_state.system_message = system_prompt
                        
                        # Clear chat and start with opening message
                        opening_message = selected_condition.get('opening_message', "How can I help you today?")
                        st.session_state.messages = [{"role": "assistant", "content": opening_message}]
                        st.session_state.usage_stats = []
                        
                        # Save selected experiment and condition
                        st.session_state.selected_experiment = experiment_names[exp_index]
                        st.session_state.selected_condition = condition_names[cond_index]
                        
                        st.success(f"Loaded: {experiment_names[exp_index]} - {condition_names[cond_index]}")
                        st.rerun()
            else:
                st.warning("Selected experiment has no conditions.")
    else:
        st.warning("No experiment files found in the 'prompts' directory.")
    
    # Chat history viewer and other sidebar elements
    st.markdown("---")
    
    # Chat history viewer
    with st.expander("View Chat History"):
        st.json(st.session_state.messages)
    
    # Usage statistics viewer
    with st.expander("View Usage Statistics"):
        if st.session_state.usage_stats:
            for i, usage in enumerate(st.session_state.usage_stats):
                st.write(f"Message {i+1}:")
                st.write(f"- Prompt tokens: {usage['prompt_tokens']}")
                st.write(f"- Completion tokens: {usage['completion_tokens']}")
                st.write(f"- Total tokens: {usage['total_tokens']}")
                st.divider()
            
            # Calculate total usage
            total_prompt = sum(u["prompt_tokens"] for u in st.session_state.usage_stats)
            total_completion = sum(u["completion_tokens"] for u in st.session_state.usage_stats)
            total = sum(u["total_tokens"] for u in st.session_state.usage_stats)
            
            st.write("### Total Usage")
            st.write(f"- Total prompt tokens: {total_prompt}")
            st.write(f"- Total completion tokens: {total_completion}")
            st.write(f"- Total tokens: {total}")
        else:
            st.write("No usage data available yet.")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]
        st.session_state.usage_stats = []
        st.success("Chat history cleared!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate and display response
    generate_response(prompt, st.session_state.system_message)