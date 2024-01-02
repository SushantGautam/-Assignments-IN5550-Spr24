import openai
import tiktoken
import ast
from datetime import datetime

# Define your OpenAI API key
api_key = json.load(open('tmp/openai.json'))["api-key"]
openai.api_key = api_key

conversation1 = lambda caption:  [
                            {
"role": "system",
"content":
    "You will play two roles: a human asking questions related to describing a short soccer video clip and "
    "an intelligent chatbot designed for video description, storytelling and captioning. "
    "Your task is to generate a detailed and descriptive paragraph based on the provided fragmented information about a short video clip. "
    "------"
    "##TASK:"
    "Users will provide event description, supporting caption and commentary of a clip, and you will generate ONE conversation-like question and answer related to describing the video and the game event in detail. "
    "The question should ask to describe the video content in detail. "
    "The answer should be a paraphrased and well-structured paragraph based on the provided description, with a minimum of 250 words and a maximum of 300 words. "
    "When the provided information is short, aim for a 250-word description, and when the provided information is more detailed, aim for very long descriptions upto 300-word description. "
    "------"
    "##INSTRUCTIONS:"
    "- The question must be like a human conversation and focused on describing the video and event in detail. "
    "- Reject the information in supporting commentary and caption if not relevant and logical to the event visible in the clip. "
    "- The answer must be a paraphrased version of the provided information, very detailed and descriptive, and within the specified word count. "
    "- Act as if you are really seeing the visual content live and have no access to the commentary and caption. Dont mention about 'commentary' and  'caption' in the answer. "
    "- Only Use the supporting commentary and caption to be smart enough to interpret the visual content, faking as though you got the information from the video itself."
    "- Avoid mentioning actual player names and team names from the commentary as it is not visible in video; instead, refer to them by jersey-color if possible, else ignore the information."
    "- Begin answers with creative opening."
},
{
"role": "user",
"content":
    f"The fragmented information: {caption}. "
    "Please generate the response in the form of a Python JSON dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
    "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
    "Emphasize that the answer should focus on describing the video content as detailed as possible."
}
]

conversation2 = lambda caption:  [
                    {
"role": "system",
"content":
    "You will play two roles: a human asking questions related to describing a short soccer video clip and "
    "an intelligent chatbot designed for video description, storytelling and captioning. "
    "Your task is to generate a detailed and descriptive paragraph based on the provided fragmented information about a short video clip. "
    "------"
    "##TASK:"
    "Users will provide multiple information from a video clip, and you will generate ONE conversation-like question and answer related to describing the video and game events in detail. "
    "The question should ask to describe the video content in detail. "
    "The answer should be a paraphrased and well-structured paragraph based on the provided information with a minimum of 250 words and a maximum of 300 words. "
    "When the provided information is short, aim for a 250-word description, and when the provided information is more detailed, aim for very long descriptions upto 300-words. "
    "------"
    "##INSTRUCTIONS:"
    "- Act as if you are only seeing the visual content live and have no access to the spporting commentary and caption text. "
    "- The question must be like a human conversation and focused on describing the video and events in detail. "
    "- Don't mention about 'commentary' and  'caption' in the answer. Fuse the information from them in answer as if you got the information from the video itself. "
    "- The answer must be a paraphrased version of the provided information, very detailed and descriptive, and within the specified word count. "
    "- Always refer to player and teams by jersey-color in answer. "
    "- Never mention real player and team names in answer, at any cost, as name is not identifiable in the clip.  Ignore the information if player or team is not resolvable to jersey-color. "
    "- Begin answers with creative opening. Don't mention about tht actual score or time as it is not visible in the clip. "
},
{
"role": "user",
"content":
    f"The fragmented video description is: {caption}.\n"
    "Please generate the response in the form of a Python JSON dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
    "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
    "Emphasize that the answer should focus on describing the video content as detailed as possible."
}
]


def call_gpt4(caption, event_n=1):
    if event_n == 1:
        conversation = conversation1(caption)
    elif event_n == 2:
        conversation = conversation2(caption)
    else:
        raise ValueError("event_n must be 1 or 2")
    print(conversation)
    
    # Encode the conversation using the specified model (e.g., "gpt-4")
    model_name = "gpt-3.5-turbo"
    encoder = tiktoken.encoding_for_model(model_name)
    encoded_text = encoder.encode("\n".join([message["content"] for message in conversation]))
    token_count = len(encoded_text)
    max_tokens = 4096
    print(token_count, "tokens inputted.")
    # return conversation


    # Check if the token count exceeds the maximum limit (4096 tokens for gpt-3.5-turbo)
    max_tokens = 4096

    if token_count <= max_tokens:
        # Make an API call to generate a response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            response_format={ "type": "json_object" },
            messages=conversation,
            # max_tokens=50  # Set the desired maximum token length for the response
        )

        # Extract and print the assistant's reply
        reply = response.choices[0].message["content"]
        # calculate token of reply
        encoded_text = encoder.encode(reply)
        token_count_reply = len(encoded_text)
        print(token_count_reply, "tokens outputted.")
        print("Assistant: gpt_response_json=", reply)
        response_dict = ast.literal_eval(reply)
        print('total tokens:', token_count + token_count_reply)
        # append to csv file, time only, input, output, inout tokens, output tokens
        # file name is date today
        log_file = f"analytics/gpt4-analytics{ datetime.now().strftime('%Y-%m-%d') }.csv"
        with open(log_file, 'a') as f:
            caption_= caption.replace('\n','\\n')
            f.write(f"{datetime.now().strftime('%H:%M:%S')},{caption_},{response_dict},{token_count},{token_count_reply},{token_count + token_count_reply}\n")
        return response_dict
    else:
        print("Token count exceeds the maximum limit.")
