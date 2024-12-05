#import dependencies
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
import anthropic
import time
import requests
import random
import ast
from IPython.display import display, HTML
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def analyze_team_guesses(data, team):
    total_expected_guesses = 0
    total_correct_guesses = 0
    hint_count = 0
    hint_found = False
    
    for line in data:
        spymaster_phrase = f'Team {team} spymaster said:'
        guess_phrase = f'Team {team} said:'
        
        if line.startswith(spymaster_phrase):
            # Extract expected guesses for the hint
            hint_found = True
            expected_guesses = int(line.split('(')[-1].split(')')[0])
            total_expected_guesses += expected_guesses
            hint_count += 1
            correct_guesses = 0  # Reset correct guesses for the new hint
        
        elif hint_found and line.startswith(guess_phrase):
            # Count correct guesses belonging to the team
            if f"The word was {team.upper()}." in line:
                correct_guesses += 1
        
        # End hint tracking when another spymaster's hint begins
        if hint_found and line.startswith("Team") and spymaster_phrase not in line:
            total_correct_guesses += correct_guesses
            hint_found = False
    
    # Add the final correct guesses after the last hint
    if hint_found:
        total_correct_guesses += correct_guesses
    
    # Calculate averages
    avg_expected_guesses = total_expected_guesses / hint_count if hint_count > 0 else 0
    avg_correct_guesses = total_correct_guesses / hint_count if hint_count > 0 else 0
    
    return {
        "team": team,
        "average_expected_guesses": avg_expected_guesses,
        "average_correct_guesses": avg_correct_guesses,
        "total_hints": hint_count
    }

"""BOARD UTILS FUNCTIONS"""

#BOARD GENERATOR
def generate_board(n = 25, lang = "eng", c = 7, k = 1):
  '''
  inputs:
    n (int): number of words
    lang (str): language of words
    c (int): number of colored words for each team
    k (int): number of killer words

  returns a dictionary such that dict[word] = color
  '''
  blue_cards = c
  red_cards = c + 1
  if (2*c + k + 1) > n:
    raise Exception("ERROR: wrong parameters for board creation")

  url = f"https://raw.githubusercontent.com/mich1803/Codenames-LLM/main/wordlists/{lang}.txt"

  try:
    # Fetch the content of the file
    response = requests.get(url)
  except:
    raise Exception("ERROR: The language entered is wrong or has no dictionary in the github folder")
  words = response.text.splitlines()

  # Select n unique random words
  random_words = random.sample(words, n)
  random_words = [word.upper() for word in random_words]

  # Select indices for the colors
  indices = list(range(n))
  blue_indices = random.sample(indices, blue_cards)
  remaining_indices = [i for i in indices if i not in blue_indices]
  red_indices = random.sample(remaining_indices, red_cards)
  remaining_indices = [i for i in remaining_indices if i not in red_indices]
  black_index = random.sample(remaining_indices, k)

  colors = [
      "BLUE" if i in blue_indices else
      "RED" if i in red_indices else
      "KILLER" if i in black_index else
      "NEUTRAL"
      for i in range(n)
      ]

  word_color_dict = {random_words[i]: colors[i] for i in range(n)}

  return word_color_dict


#BOARD FORMATTING FOR PROMPT
def board4prompt(word_color_dict, master = False):
  '''
  inputs:
    word_color_dict(dict): board dict generated from generate_board()
    matser(bool): tells if you need to see the color on the board

  return a string version for the prompt for LLMs
  '''

  text = "|"
  if master:
    for idx, i in enumerate(word_color_dict):
      text += f" {i} ({word_color_dict[i]}) " + "|"
      if (idx > 0) and (((idx + 1) % 5) == 0):
        text += "\n"
        if idx != (len(word_color_dict)-1):
          text += "|"
  else:
    for idx, i in enumerate(word_color_dict):
      text += f" {i} " + "|"
      if (idx > 0) and ((idx % 5) == 0):
        text += "\n"
        if idx != (len(word_color_dict)-1):
          text += "|"


  return text


#CARD IMAGES
def create_image_dict(word_color_dict, master = False):
  """
  inputs:
    word_color_dict(dict): board dict generated from generate_board()
    master(bool): tells if you need to see the color on the board

  returns a dict that maps each key to his card image
  """

  #DICT WITH CARD IMAGES LINK
  neutral_links = [f"https://github.com/mich1803/Codenames-LLM/blob/main/graphics/NEUTRAL{i}.jpg?raw=true" for i in range(1, 5)]
  blue_links = [f"https://github.com/mich1803/Codenames-LLM/blob/main/graphics/BLUE{i}.jpg?raw=true" for i in range(1, 5)]
  red_links = [f"https://github.com/mich1803/Codenames-LLM/blob/main/graphics/RED{i}.jpg?raw=true" for i in range(1, 5)]
  team_links = {"RED": red_links, "BLUE": blue_links}
  killer_link = "https://github.com/mich1803/Codenames-LLM/blob/main/graphics/KILLER.jpg?raw=true"
  neutral_link = "https://github.com/mich1803/Codenames-LLM/blob/main/graphics/neutral.jpg?raw=true"

  image_dict = {}
  if master:
    for word in word_color_dict:
      if word_color_dict[word] == "NEUTRAL":
        image_dict[word] = random.choice(neutral_links)
      elif word_color_dict[word] == "BLUE":
        image_dict[word] = random.choice(team_links["BLUE"])
      elif word_color_dict[word] == "RED":
        image_dict[word] = random.choice(team_links["RED"])
      elif word_color_dict[word] == "KILLER":
        image_dict[word] = killer_link

  else:
    for word in word_color_dict:
        image_dict[word] = neutral_link

  return image_dict


#STYLISH BOARD FORMATTING FOR PRINT
def board4print(image_dict):
    """
    Inputs:
        image_dict (dict): A dictionary where keys are words, and values are image links.

    Returns:
        str: A formatted HTML string for rendering the game board.
    """
    if not image_dict:
        raise ValueError("The image_dict is empty or invalid.")

    # Start HTML table
    html = """<table style="border-collapse: collapse; margin: 0 auto; width: 60%; text-align: center;">"""

    # Split words into rows (5 columns per row)
    words = list(image_dict.keys())
    rows = [words[i:i + 5] for i in range(0, len(words), 5)]

    for row in rows:
        html += "<tr>"
        for word in row:
            image_link = image_dict[word]
            html += f"""
            <td style="
                font-size: 30px;
                background-image: url('{image_link}');
                background-size: cover;
                background-position: center;
                color: white;
                padding: 10px;
                border-radius: 0px;
                border: 5px solid white; /* Add white border */
                width: 20%;
                height: 100px;
                text-shadow: 3px 3px #171717;
                overflow: hidden; /* Ensures border-radius applies */
            ">
                <b>{word}</b>
            </td>
            """
        html += "</tr>"

    html += "</table>"
    return html



"""API AGENTS"""

def call_api(system_prompt, prompt, model, json_mode):
  """
  inputs:
    system_prompt(str): system prompt for the api call
    prompt(str): prompt for the api call
    model(str)
    json_mode(bool): formatting style json

  return the output of the api call
  """
  load_dotenv()
  API_providers = ["GOOGLE", "OPENAI", "XAI", "LLAMA", "ANTHROPIC"]
  API_KEY = {p: os.getenv(f"{p}_key")  for p in API_providers}


  #OPENAI models:
  if model in ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-o1-mini", "gpt-o1-preview"): 
    client = OpenAI(api_key=API_KEY["OPENAI"])

    #OPENAI normal mode
    if not json_mode:
      chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                      {"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}
                      ]
        )
      return chat_completion.choices[0].message.content

    #OPENAI json mode
    if json_mode:
      chat_completion = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}
                  ]
        )
      answer = chat_completion.choices[0].message.content
      try:
        data = ast.literal_eval(answer)

      except (SyntaxError, ValueError) as e:
        # Catch specific errors and provide more context in the error message
        print(f"Error parsing API response: {e}")
        print(f"Response content: {answer}")
        raise ValueError("Format error: The API response is not a valid Python literal.") from e  # Chain exceptions for better debugging
      return data


  #XAI models
  elif model in ("grok-beta"):
    client = OpenAI(api_key=API_KEY["XAI"], base_url="https://api.x.ai/v1")

    #XAI normal mode
    if not json_mode:
      chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                      {"role": "system", "content": system_prompt + 'Dont format the code, GIVE ONLY THE RAW DICT'},
                      {"role": "user", "content": prompt}
                      ]
        )
      return chat_completion.choices[0].message.content

    #XAI json mode
    if json_mode:
      chat_completion = client.chat.completions.create(
        model=model,
        messages=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": prompt}
                  ]
        )
      answer = chat_completion.choices[0].message.content
      answer = answer.removeprefix('```json')
      answer = answer.removeprefix('```python')
      answer = answer.removesuffix('```')
      try:
        data = ast.literal_eval(answer)

      except (SyntaxError, ValueError) as e:
        # Catch specific errors and provide more context in the error message
        print(f"Error parsing API response: {e}")
        print(f"Response content: {answer}")
        raise ValueError("Format error: The API response is not a valid Python literal.") from e  # Chain exceptions for better debugging
      return data


  #ANTHROPIC models
  elif model in ("claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"):
    client = anthropic.Anthropic(api_key=API_KEY["ANTHROPIC"])

    #ANTHROPIC json mode
    if json_mode:
      message = client.messages.create(
          model=model,
          max_tokens=1000,
          system=system_prompt + "This response is going to be read as a python ast literal, format the text as it is. don't start new paragraph etc.",
          messages=[
              {
                  "role": "user",
                  "content": [
                      {
                          "type": "text",
                          "text": prompt
                      }
                  ]
              }
          ]
      )
      answer = message.content[0].text
      answer = answer.removeprefix('```json')
      answer = answer.removeprefix('```python')
      answer = answer.removesuffix('```')
      try:
        data = ast.literal_eval(answer)

      except (SyntaxError, ValueError) as e:
        # Catch specific errors and provide more context in the error message
        print(f"Error parsing API response: {e}")
        print(f"Response content: {answer}")
        raise ValueError("Format error: The API response is not a valid Python literal.") from e  # Chain exceptions for better debugging
      return data

    #ANTHROPIC normal mode
    else:
      message = client.messages.create(
          model=model,
          max_tokens=1500,
          system=system_prompt,
          messages=[
              {
                  "role": "user",
                  "content": [
                      {
                          "type": "text",
                          "text": prompt
                      }
                  ]
              }
          ]
      )
      return message.content[0].text


  #LLAMA API MODELS
  elif model in ("llama3.2-1b", "llama3.2-11b-vision", "llama3.2-90b-vision"):
    client = OpenAI(api_key=API_KEY["LLAMA"], base_url="https://api.llama-api.com")

    #LLAMA normal mode
    if not json_mode:
      chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                      {"role": "system", "content": system_prompt},
                      {"role": "user", "content": prompt}
                      ]
        )
      return chat_completion.choices[0].message.content

    #LLAMA json mode
    if json_mode:
      while True:
        dict_request = "GIVE THE ANSWER AS A DICTIONARY \n example: \n {'thought': 'insert your thought', 'clue':'insert your clue', 'number':'insert the number'}" +\
        "or with the other keys that I asked you. This response is going to be read as a python literal."
        chat_completion = client.chat.completions.create(
          model=model,
          response_format={ "type": "json_object" },
          max_tokens=1500,
          messages=[
                    {"role": "system", "content": system_prompt + dict_request},
                    {"role": "user", "content": prompt}
                    ]
          )
        answer = chat_completion.choices[0].message.content
        answer = answer.removeprefix('```json')
        answer = answer.removeprefix('```python')
        answer = answer.removesuffix('```')
        try:
          data = ast.literal_eval(answer)
          break
        except (SyntaxError, ValueError) as e:
          pass
      return data


  elif model in ("gemini-1.5-flash-latest", "gemini-1.5-pro-latest"):
    genai.configure(api_key=API_KEY["GOOGLE"])
    genmodel = genai.GenerativeModel(model)
    if json_mode:
      while True:
        #time.sleep(4)
        answer = genmodel.generate_content(system_prompt + "This response is going to be read as a python ast literal, format the text as it is. don't start new paragraph etc." + prompt).text
        answer = answer.removeprefix('```json')
        answer = answer.removeprefix('```python')
        answer = answer[:-1]
        for u in range(3):
          answer = answer.removesuffix('`')
        try:
          data = ast.literal_eval(answer)
          break

        except:
          pass

      return data
    else:
      answer = genmodel.generate_content(system_prompt + prompt).text
      return answer

  else:
    raise Exception("ERROR: wrong model name")


#SPYMASTER PROMPTS to API CALL
def spymaster(lang, board, team, history, model, cot):
  """
  inputs:
    lang(str): language
    board(str): formatted board for prompt
    team(str): team of the spymaster called
    history(list[str]): past history of the game
    model(str): 
    cot(bool): chain of tought ativation

  return clue and number from the spymaster
  """
  json_mode = True
  sys_prompt = (
      "We are playing the board game CODENAMES, and your role is the spymaster."+ \
      f"You are part of team {team}, and your job is to help your team guess the words associated with your team's color."+ \
      "You will be given a list of words, each with an associated color: RED for team red words, BLUE for team blue words, "+ \
      "KILLER for killer words, and NEUTRAL for neutral words."+ \
      "Your goal is to provide a ONE-WORD CLUE and a NUMBER. The clue should help your team guess only the words with your team's color."+ \
      "If your team guesses the correct word, you gain 1 point. If your team guesses an opponent's word, they gain 1 point. "+ \
      "If your team guesses the KILLER word, you lose instantly, so be extremely cautious!"+ \
      "Be careful with your clues! Important: DO NOT use any words that are already on the board, as that would immediately make you lose."+ \
      "To ensure you do not accidentally break this rule, follow these steps strictly:"+ \
      "1. Before generating a clue, check the list of words on the board. If the clue you are about to give matches any word on the board, discard it and choose another clue."+ \
      "2. Use synonyms, related concepts, or descriptive ideas to create your clue, ensuring it is not an exact match for any word on the board."+ \
      "3. Double-check your clue against all words on the board before finalizing your response."+ \
      "Your clue should not be too vague or too specific. The clue should help your team make a guess about the words on the board, but it should be safe and avo+ \id leading to mistakes."+ \
      "You should also be mindful of these important factors:"+ \
      "1. Avoid using a word from the board as your clue. If you do, you will lose instantly."+ \
      "2. The clue should not be misleading or overly broad. Think of how it could be interpreted and avoid ambiguities."+ \
      "3. The number you provide should represent how many words on the board are related to your clue."+ \
      "4. The number should be chosen carefully. Do not overestimate how many words fit the clue, and avoid making the number too high, "+ \
      "as it could increase the risk of the guessers making mistakes."+ \
      "5. If your team guesses the KILLER word, you will lose the game."+ \
      "6. Consider the current state of the game: If there are few words left, give more specific clues. If there are many words left, "+ \
      "give broader clues that point to multiple words."+ \
      "7. Always prioritize safety. If a clue could be interpreted in a way that risks your team guessing incorrectly, choose a safer clue."+ \
      "After you choose the clue, double-check if it’s not in the board and strategically helpful for your team.".upper()
)

  if not cot:
    adding_cot = "YOUR RESPONSE SHOULD BE A JSON PYTHON DICT OBJECT WITH TWO KEYS: 'clue' and 'number'. Make sure the clue is a single word, and the number "+ \
      "is an integer between 1 and the number of words left on the board related to that clue."
  else:
    adding_cot = "YOUR RESPONSE SHOULD BE A JSON PYTHON DICT OBJECT WITH THREE KEYS: 'thought', 'clue' and 'number'. Make sure the clue is a single word, and the number "+ \
      "is an integer between 1 and the number of words left on the board related to that clue." +\
      "In the thought (the fisrt thing you are going to say as a 200 words max test) you need to use reasoning to understand which clue and number to say, divide the problem in subproblem like:" +\
      "- which words i cannot say like the ones in the board (or you lose instantly);" +\
      f"- which are the words of your team ({team}) and which word can I say that aligns with them;" +\
      "- which are the words the you are better to avoid make your team think, like the ones of the opponents' team or the killer one." +\
      "at the end double check that your clue is not in the board to avoid losing instantly.".upper() +\
      "BE EXTREMELY SURE THAT THE DICT KEYS ARE RIGHT (thought, clue, number)."

  sys_prompt += adding_cot

  prompt = f"The board is {board4prompt(board, master = True)}, The history of the game is: {history}."

  response = call_api(sys_prompt, prompt, model, json_mode)
  #if cot: print(response["thought"])
  return response["clue"].upper(), int(response["number"])

#GUESSER PROMPTS TO API CALL
def guesser(lang, team, board, clue, n_guessers, i_agent, cards_remaining, idea, k, history, conv, model):
  """
  inputs:
    lang(str): language
    team(str): team
    clue(str): clue from the spymaster
    board(str): formatted board for prompt
    history(list[str]): past history of the game
    n_guessers(int): number of guessers
    i_agent(int): number of the agent called
    cards_remaining(int): number of cards remaining
    idea(str): the previous thoughts of the guesser called
    k(int): number of killer cards in the board
    conv(str): past conversation
    model(str): 

  return conversation, want_to_talk and vote
  """
  opp = "BLUE" if team == "RED" else "RED"
  json_mode = True
  sys_prompt = "We are playing the board game CODENAMES, and your role" + \
              f"will be the guesser. \n You're in a team of {n_guessers} (BUT TALK AS ONLY ONE OF YOU, don't talk for the others)" +\
              f"other guessers. Your objective as a team is to guess" +\
              f"the words of your color based on the clue of your spymaster \n" +\
              f"There are: {cards_remaining[team]} words for your team, "+\
              f"{cards_remaining[opp]} words for the enemy team and {k} killer cards." + \
              f"Even if the number said by the spymaster is more than one, let's guess one word at a time. \n" +\
              f"You are in team {team}. You are agent number {i_agent}." +\
              f"Your inital thoughts were: {idea} \n" +\
              f"You are a very good JSON file writer." +\
              f"You need to give in output a PYTHON JSON DICT OBJECT with 3 keys:\n" +\
              f"M: your message to other teammates (be short and coincise); \n" +\
              f"W: a bool that is 1 if you want to listen and speak to the others again, 0 if you feel like you wouldn't add anything to the conversation (*if you all agree please say 0*); \n" +\
              f"V: a dictionary that maps each word in the board to a real number in iterval (-10,10) that represents your confidence in guessing that word (don't be afraid of writing decimal numbers)."

  prompt = f"The board is {board4prompt(board)}. \n" +\
          f"The history is {history}. \n" +\
          f"The clue is: {clue}. \n" +\
          f"The previous conversation is: {conv}."

  response = call_api(sys_prompt, prompt, model, json_mode)
  #print(response["W"])
  return response["M"], bool(response["W"]), response["V"]


def solo_guesser(lang, team, board, clue, cards_remaining, k, history, model, cot):
  """
  inputs:
    ...

  return vote
  """
  opp = "BLUE" if team == "RED" else "RED"
  json_mode = True
  sys_prompt = "We are playing the board game CODENAMES, and your role" + \
              f"will be the guesser." +\
              f"Your objective as a team is to guess" +\
              f"the words of your color based on the clue of your spymaster \n" +\
              f"There are: {cards_remaining[team]} words for your team, "+\
              f"{cards_remaining[opp]} words for the enemy team and {k} killer cards." + \
              f"Even if the number said by the spymaster is more than one, let's guess one word at a time. \n" +\
              f"You are in team {team}." +\
              f"You are a very good JSON file writer."
  if cot:
    cot_adding = f"You need to give in output a PYTHON JSON DICT OBJECT with 2 keys:\n" +\
              f"T: your your inner thought (think out loud and break the solution in subproblems); \n" +\
              f"W: the word you are going to guess."
  else:
    cot_adding = f"You need to give in output a PYTHON JSON DICT OBJECT with 1 key:\n" +\
              f"W: the word you are going to guess."

  prompt = f"The board is {board4prompt(board)}. \n" +\
          f"The history is {history}. \n" +\
          f"The clue is: {clue}. \n"

  response = call_api(sys_prompt + cot_adding, prompt, model, json_mode)
  #print(response["W"])
  return response["W"]


#GUESSER PROMPTS to API CALL
def guesser_ideas(lang, team, board, clue, n_guessers, i_agent, cards_remaining, k, history,  model = "gpt-4o-mini"):
  """
  inputs:
    lang(str): language
    team(str): team
    clue(str): clue from the spymaster
    board(str): formatted board for prompt
    history(list[str]): past history of the game
    n_guessers(int): number of guessers
    i_agent(int): number of the agent called
    cards_remaining(int): number of cards remaining
    k(int): number of killer cards in the board
    model(str): 

  return the initial idea of the guesser
  """
  opp = "BLUE" if team == "RED" else "RED"
  json_mode = False
  sys_prompt = "We are playing the board game CODENAMES, and your role" + \
              f"will be the guesser. \n You're in a team of {n_guessers}" +\
              f"other guessers. Your objective as a team is to guess" +\
              f"the words of your color based on the clue of your spymaster \n" +\
              f"There are: {cards_remaining[team]} words for your team, "+\
              f"{cards_remaining[opp]} words for the enemy team and {k} killer cards." + \
              f"Even if the number said by the spymaster is more than one, let's guess one word at a time. \n" +\
              f"You are in team {team}. You are agent number {i_agent}." +\
              "Try to say a max of 200 characters."

  prompt = f"The board is {board4prompt(board)}. \n" +\
          f"The history is {history}. \n" +\
          f"The clue is: {clue}. \n" +\
          "Share your (short and coincise) initial thoughts with your other fellow teammates."

  response = call_api(sys_prompt, prompt, model, json_mode)
  return response


#VOTE SYSTEM
def vote_system(votes):
  """
  inputs:
    votes(list[dict]): votes from the guessers

  return the guess of the team after the vote
  """
  results = {}
  for vote in votes:
    for word, points in vote.items():
      if word in results:
        results[word] += points
      else:
        results[word] = points
  return max(results, key=results.get)


#MAIL FUNCTION (HUMAN API CALL XD)



"""GAME FUNCTIONS"""
def send_email(recipient_email, html_board):
    """Sends an email with the HTML board to the specified recipient.

    Args:
        recipient_email: The email address of the recipient.
        html_board: The HTML content of the board.
        sender_email: Your email address (default: your_email@gmail.com).
                      **Replace with your actual email address.**
        sender_password: Your email password (default: your_password).
                         **Replace with your actual password or an app password.**
    """
    sender_email = os.getenv("MAIL")
    sender_password = os.getenv("MAIL_apppsw")
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "Codenames Board"

    msg.attach(MIMEText(html_board, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("Master board sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

#TURN FUNCTION
def play_turn(lang, team, board, cards_remaining, k, n_guessers, history, image_dict, master_image_dict, master_model, guesser_model, cot, cot_guesser, verbose = False, masterverbose = False):
  """
  Simulates a turn for a team in a word-based guessing game.

    Inputs:
    - lang (str): The language of the game (e.g., "EN" for English).
    - team (str): The team taking the turn ("RED" or "BLUE").
    - board (dict): The current game board, mapping words to roles ("RED", "BLUE", "NEUTRAL", or "KILLER").
    - cards_remaining (dict): Remaining words to guess for each team (e.g., {'RED': 5, 'BLUE': 6}).
    - k (int): Parameter influencing AI decision-making.
    - n_guessers (int): Number of guessers (if using AI guessers).
    - history (list): A log of all clues and guesses so far.
    - image_dict (dict): Maps guessed words to images (for visualization).
    - master_image_dict (dict): Original image mapping for all board words.
    - model (str): Mode of play:
    - "human": Clues and guesses are provided manually.
    - Other values: AI-controlled gameplay.
    - cot (bool): Enables chain-of-thought reasoning for AI models.
    - verbose (bool): If True, prints detailed game status.
    - masterverbose (bool): If True, displays the full board for the spymaster.

    Outputs:
    Returns a dictionary summarizing the turn outcome:
        If the game ends:
        - "endgame" (bool): True.
        - "win" (str): The winning team ("RED" or "BLUE").
        - "reason" (str): Why the game ended ("killer word selected", "cards finished").
        - "history" (list): Updated log of the game history.
        If the game continues:
        - "endgame" (bool): False.
        - "rc" (dict): Updated remaining cards for each team.
        - "b" (dict): Updated game board.
        - "spyh" (str): Spymaster's clue and number.
        - "teamh" (list): The team's guesses and their outcomes.

  """

  #DICT WITH PROMPT COLORS
  prompt_colors = {
      "RED": "\033[1;31m",
      "BLUE": "\033[1;34m",
      "endcolor": "\033[0m",
      "n": "\033[1;33m"
    }


  opp = "BLUE" if team == "RED" else "RED"
  rc = cards_remaining
  b = board
  if verbose:
    print("The actual board is: ")
    if masterverbose:
      display(HTML(board4print(master_image_dict)))
    else:
      display(HTML(board4print(image_dict)))
    print(f"it is {prompt_colors[team]}{team}{prompt_colors['endcolor']} team turn: \n")


  n_clue_board = 0
  cib_control = True # Clue in board control variable
  while cib_control:
    if master_model == "human":
      clue = input("Insert a clue for your teammates:")
      number = input("Insert the number of words related to that clue: ")
    else:
      clue, number = spymaster(lang, board, team, history, master_model, cot)
    spymaster_history = f"Team {team} spymaster said: {clue} ({number})."
    guesser_history = []
    if verbose: print(f"The {prompt_colors[team]}{team}{prompt_colors['endcolor']} spymaster's clue is: {clue} ({number}). \n \n")

    if clue in b:
      if verbose: print(f"{clue} is in the board, try again. \n \n")
      n_clue_board += 1
    else:
        cib_control = False


  for i in range(number):
    if guesser_model == "model":
      guess = input("Take your guess")
    else:
      ideas = []
      votes = [{} for _ in range(n_guessers)]
      want_to_talk = [True for _ in range(n_guessers)]
      conv = []

      if n_guessers > 1:
        for j in range(n_guessers):
          ideas.append(guesser_ideas(lang, team, board, clue, n_guessers, j, cards_remaining, k, history,  guesser_model))
          if verbose: print(f"Team {prompt_colors[team]}{team}{prompt_colors['endcolor']} guesser {j} thinks: {ideas[j]}.")
          conv.append(f"Team {team} guesser {j} said: {ideas[j]}.")
        speaktoomuch = 0
        while (any(want_to_talk)) and speaktoomuch < 2:
          if verbose: print(f"Do you want to talk? {want_to_talk}")
          for j in range(n_guessers):
            if want_to_talk[j]:
              mess, want_to_talk[j], votes[j] = guesser(lang, team, board, clue, n_guessers, j, cards_remaining, ideas[j], k, history, conv, guesser_model)
              if verbose: print(f"Team {prompt_colors[team]}{team}{prompt_colors['endcolor']} guesser {j} said: {mess}.")
              conv.append(f"Team {team} guesser {j} said: {mess}.")
          speaktoomuch += 1
        guess = vote_system(votes)
        if verbose: print(f"{prompt_colors['n']}NARRATOR{prompt_colors['endcolor']}: Team {prompt_colors[team]}{team}{prompt_colors['endcolor']} voted: {guess}. \n \n")
    
      elif n_guessers == 1:
          guess = solo_guesser(lang, team, board, clue, cards_remaining, k, history, guesser_model, cot_guesser)
          if verbose: print(f"{prompt_colors[team]}{team} GUESSER{prompt_colors['endcolor']}: {guess}. \n \n")
      

    try:
        x = b[guess]
    except:
        if verbose: print(f"{prompt_colors[team]}{team}{prompt_colors['endcolor']} team guess wasn't on the board. \n \n")
        guesser_history.append(f"Team {team} said: {guess}. The word was not in board")
        return {"endgame": False, "rc": rc, "b": b, "spyh": spymaster_history, "teamh": guesser_history, "cib": n_clue_board}

    image_dict[guess] = master_image_dict[guess]
    del master_image_dict[guess]
    if x == "KILLER":
        if verbose: print(f"The killer word have been selected, the game ends. \n \n ")
        guesser_history.append(f"Team {team} said: {guess}. The word was {x}.")
        history.append(spymaster_history)  # Append spymaster_history
        history.extend(guesser_history)  # Extend with guesser_history
        return {"endgame": True, "win": opp, "reason": "killer word selected", "history": history, "cib": n_clue_board}

    elif x == team:
        rc[team] -= 1
        if verbose: print(f'A {prompt_colors[team]}{team} word{prompt_colors["endcolor"]} have been selected ({team} cards remaining = {rc[team]}). \n \n')
        del b[guess]
        guesser_history.append(f"Team {team} said: {guess}. The word was {team}.")
        if rc[team] == 0:
            if verbose: print(f"The {prompt_colors[team]}{team} team{prompt_colors['endcolor']} reached the goal, the game ends. \n \n")
            history.append(spymaster_history)  # Append spymaster_history
            history.extend(guesser_history)  # Extend with guesser_history
            return {"endgame": True, "win": team, "reason": "cards finished", "history": history, "cib": n_clue_board}

    elif x == opp:
        rc[opp] -= 1
        if verbose: print(f'A {prompt_colors[opp]}{opp} word{prompt_colors["endcolor"]} have been selected ({opp} cards remaining = {rc[opp]}). \n \n')
        del b[guess]
        guesser_history.append(f"Team {team} said: {guess}. The word was {opp}.")
        if rc[opp] == 0:
            if verbose: print(f"The {prompt_colors[opp]}{opp} team{prompt_colors['endcolor']} reached the goal, the game ends. \n \n")
            history.append(spymaster_history)  # Append spymaster_history
            history.extend(guesser_history)  # Extend with guesser_history
            return {"endgame": True, "win": team, "reason": "cards finished", "history": history, "cib": n_clue_board}
        break

    else:
        guesser_history.append(f"Team {team} said: {guess}. The word was neutral.")
        if verbose: print(f"A neutral word have been selected. \n \n")
        del b[guess]
        break

  return {"endgame": False, "rc": rc, "b": b, "spyh": spymaster_history, "teamh": guesser_history, "cib": n_clue_board}


#GAME FUNCTION
def play_game(lang = "eng", n_cards = 25, coloured_cards = 7, k_cards = 1, verbose = False, red_master_model = "gpt-4o-mini", red_guesser_model = False, red_cot = True, red_cot_guesser = False, blue_master_model = "gpt-4o-mini", blue_guesser_model = False, blue_cot = True, blue_cot_guesser = False, red_agents = 3, blue_agents = 3, masterverbose = False):
  """
  Simulates a full game of a word-based guessing game, alternating turns between two teams until a winning condition is met.

    Inputs:
    - lang (str): The language of the game (default: "eng").
    - n_cards (int): Total number of cards on the board (default: 25).
    - coloured_cards (int): Number of cards assigned to each team's color (default: 7).
    - k_cards (int): Number of killer cards on the board (default: 1).
    - verbose (bool): If True, displays detailed game status and parameters (default: False).
    - red_model (str): The model used by the RED team spymaster ("human" or AI model name, default: "gpt-4o-mini").
    - red_cot (bool): Enables chain-of-thought reasoning for the RED team model (default: True).
    - red_cot_guesser (bool): Enables chain-of-thought reasoning for the RED guessers (if 1) (default: False).
    - blue_model (str): The model used by the BLUE team spymaster ("human" or AI model name, default: "gpt-4o-mini").
    - blue_cot (bool): Enables chain-of-thought reasoning for the BLUE team model (default: True).
    - blue_cot_guesser (bool): Enables chain-of-thought reasoning for the BLUE guessers (if 1) (default: False).
    - red_agents (int): Number of guessing agents for the RED team (default: 3).
    - blue_agents (int): Number of guessing agents for the BLUE team (default: 3).
    - masterverbose (bool): If True, displays the complete board for spymasters during the game (default: False).

    Outputs:
    Returns a tuple summarizing the game results:
    - win (str): The winning team ("RED" or "BLUE").
    - reason (str): The reason for the game's end ("killer word selected", "cards finished").
    - r (int): The total number of rounds played.
    - history (list): A log of all clues and guesses during the game.
  """
  #DICT WITH PROMPT COLORS
  prompt_colors = {
      "RED": "\033[1;31m",
      "BLUE": "\033[1;34m",
      "endcolor": "\033[0m",
      "n": "\033[1;33m"
    }


  if not red_guesser_model: red_guesser_model = red_master_model
  if not blue_guesser_model: blue_guesser_model = blue_master_model
  if red_guesser_model == "human": red_agents = 1
  if blue_guesser_model == "human": blue_agents = 1
  r = 1
  cib = {"RED": 0, "BLUE": 0}
  turn = "RED"
  board = generate_board(n = n_cards, lang = lang, c = coloured_cards, k = k_cards)
  image_dict = create_image_dict(board, master = False)
  master_image_dict = create_image_dict(board, master = True)
  initial_board = master_image_dict.copy()
  cards_remaining = {"RED": coloured_cards + 1, "BLUE": coloured_cards}
  history = []

  if red_master_model == "human":
    mail = input("Insert the mail of the red spymaster:")
    send_email(mail, board4print(master_image_dict))
  if blue_master_model == "human":
    mail = input("Insert the mail of the blue spymaster:")
    send_email(mail, board4print(master_image_dict))
  #input("stop")

  if verbose:
      intro = prompt_colors["n"] + "GAME PARAMETERS:" + prompt_colors["endcolor"] + "\n" + \
      f"language = {lang}, \n" + \
      f"number of cards = {n_cards}, \n" + \
      f"number of killer cards = {k_cards}, \n" + \
      "\n" + \
      prompt_colors["RED"] + "RED TEAM PARAMETERS:" + prompt_colors["endcolor"] + "\n" + \
      "Red team starts first, " + \
      f"number of red cards = {cards_remaining['RED']}, \n" + \
      f"red spymaster model = {red_master_model}, \n" + \
      f"red guesser model = {red_guesser_model}, \n" + \
      f"number of red agents = {red_agents}, \n" + \
      "\n" + \
      prompt_colors["BLUE"] + "BLUE TEAM PARAMETERS:" + prompt_colors["endcolor"] + "\n" + \
      f"number of blue cards = {cards_remaining['BLUE']}, \n" + \
      f"blue spymaster model = {blue_master_model}, \n" + \
      f"blue guesser model = {blue_guesser_model}, \n" + \
      f"number of blue agents = {blue_agents}, \n"

      print("-----------------------------------")
      print(intro)
      print("-----------------------------------")

  while True:
    result = play_turn(lang = lang,
                       team = turn,
                       board = board,
                       cards_remaining = cards_remaining,
                       k = k_cards,
                       n_guessers = red_agents if turn == "RED" else blue_agents,
                       history = history,
                       image_dict = image_dict,
                       master_image_dict = master_image_dict,
                       master_model = red_master_model if turn == "RED" else blue_master_model,
                       guesser_model = red_guesser_model if turn == "RED" else blue_guesser_model,
                       cot = red_cot if turn == "RED" else blue_cot,
                       cot_guesser = red_cot_guesser if turn == "RED" else blue_cot_guesser,
                       verbose = verbose,
                       masterverbose = masterverbose)
    cib[turn] += result["cib"]
    if result["endgame"]:
      if verbose:
        print(result)
        display(HTML(board4print(initial_board)))
      return (result["win"], result["reason"], r, history, cib["RED"], cib["BLUE"])
    else:
      cards_remaining = result["rc"]
      board = result["b"]
      r += 1
      turn = "BLUE" if turn == "RED" else "RED"
      history.append(result["spyh"])
      history += result["teamh"]