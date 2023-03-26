import openai
import os

# Apply the API key
openai.api_key = "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"

input_dir = './caption/'  # Set the path to the directory containing the input text files
output_dir = './imagine/'  # Set the path to the directory to store the output text files

if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the output directory if it doesn't already exist

for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        with open(input_path, 'r') as f:
            prompt = "Assuming you are a painter and you have an unfinished artwork that needs to be completed. The center of the painting is "+f.read()+" Please imagine the remaining parts and just respond the description of the painting, do not respond any other things, and your respond should less than 20 words."
            print(prompt)
        completions = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = completions.choices[0].text
        print(message)
        with open(output_path, 'w') as f:
            f.write(message)
