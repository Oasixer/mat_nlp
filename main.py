import pyautogui as pgui
import os
import sys
import nltk
from inputs import devices
from inputs import get_key
import bs4 as bs  
import urllib.request  
import re
import heapq
from googlesearch import search

x_TL=0
y_TL=0

run_first_keyword = True

pgui.FAILSAFE = True

input_modes = ['keypress', 'stdin']
input_mode = input_modes[0]

enable_nltk = True

glossary_file_prefix='glossary_'

debug_level = -1 # Above 0 = more printing. -1 just does shit faster without telling you

def get_num(prompt="Input chapter number: "):
    print(prompt)
    while True:
        if input_mode == input_modes[1]:
            num = input()
            if num == 'q' or not num:
                # Quit
                exit()
            try:
                num = int(num)
                return num
            except:
                print("Invalid!")
        else:
            events = get_key()
            for event in events:
                if event.code[4].isdigit():
                    print(f"Chapter {event.code[4]}")
                    return int(event.code[4])
                if event.code == 'KEY_ESC' and event.state == 1:
                    print('Exit key (ESC) detected. Program quitting.')
                    exit()


def get_key_state():
    while True:
        events = get_key()
        for event in events:
            if event.code == 'KEY_LEFTCTRL' and event.state == 1:
                return 1
            if event.code == 'KEY_ESC' and event.state == 1:
                print('Exit key (ESC) detected. Program quitting.')
                exit()
            if event.code == 'KEY_LEFTSHIFT' and event.state == 1:
                return -1

def get_select():
    pgui.alert("Select text, press ctrl when selection is ready")
    get_key_state()

    text = os.popen('xsel').read()
    return text

def find_keyword_position(word):
    pgui.scroll(100)
    pgui.scroll(0)
    pgui.moveTo(x_TL, y_TL+100)
    old_text = os.popen('xsel').read()
    pgui.PAUSE=0.2
    pgui.mouseUp()
    pgui.mouseDown()

    text=old_text
    while word not in text or (text == old_text):
        pgui.mouseDown()
        text = os.popen('xsel').read()
        pgui.keyDown('down', pause=0.05)
        pgui.keyUp('down')
        
    pgui.keyUp('down')
    pgui.mouseUp()
    pgui.moveRel(50, -100, pause=0.3)
    pgui.click(pause=0.3)
    pgui.mouseDown(pause=0.3)
    pgui.dragRel(70, 50, pause=0.3, duration=1)
    pgui.mouseUp()
    print(text)

def normal_mode():
    #pgui.alert("ready? (mouse to top left of text! and use space to OK)")
    global x_TL, y_TL
    x_TL=pgui.position()[0]
    y_TL=pgui.position()[1]
    print("Normal mode selected.")
    
    chapter = 0

    pgui.mouseDown()
    pgui.scroll(-500)
    pgui.moveRel(0, 200)
    text = os.popen('xsel').read()
    pgui.mouseUp()

    m = re.search("\d", text)
    if m:
        chapter = int(text[int(m.start())])
    try:
        with open(glossary_file_prefix + str(chapter) + '.txt', 'r') as f:
            key_terms = f.read().splitlines()
            key_terms = [term[:-1] for term in key_terms]
    except FileNotFoundError:
        print("Invalid chapter number! Restarting normal mode.")
        normal_mode()
    
    if (debug_level > -1):
        print(key_terms)

    
    found = [] # Term string, index int pairs
    for term in key_terms:
        index = text.find(term)
        if (index is not -1):
            found.append([term, index])

    if not found:
        print("No key terms found")
        print("exiting")
        exit()

    i=1
    stringy = ""
    for term in found:
        stringy+= f"Key term #{i} found: " + term[0] +"\n"
        i+=1

    if (not run_first_keyword):
        query = found[int(pgui.prompt(stringy+"\n\nWhich term do you want to... 'research'?"))-1][0]
    else:
        query = found[0][0]
    for j in search(query + " wikipedia", tld="com", num=1, stop=1, pause=1): 
        link = j


    if (not run_first_keyword):
        pgui.confirm('Open link? Url: '+ link)

    scraped_data = urllib.request.urlopen(link)  
    
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article,'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:  
        article_text += p.text

    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )  
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(article_text)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}  
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]


    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)

    print(summary)

    #pgui.confirm("Ready to find the keyword in the text?")

    find_keyword_position(query)

    #pgui.keyDown('capslock')
    #pgui.keyDown('shiftleft')

    intros = [
    query + " is a very important concept for this unit. I've summarized some of the key points in the wiki article which can be found here: " + link + ""
    ]

    #pgui.confirm("")

    pgui.typewrite(intros[0] + summary, interval=0.01)

    #pgui.keyUp('capslock')
    #pgui.keyUp('shiftleft')


def parse_glossary():
    print("Glossary mode.")
    chapter = get_num()
    text = get_select().lower()
    length0 = len(text)
    while(True):
        index1 = text.find('(')
        if index1 == -1:
            break
        index2 = text.find(')')
        text = text[:index1] + text[index2+2:]
        if len(text) >= length0:
            print("Error! String is not shrinking!")
            exit()
    print(text)
    with open(glossary_file_prefix+str(chapter)+'.txt', 'w') as f:
        f.writelines(text)


inp = pgui.confirm("Normal mode: Press OK, then move mouse to top left of text (above chapter number), then press OK\nGlossary parse mode: Press cancel")
    #get_key_state()
if inp=='OK':
    #global x_TL, y_TL
    #x_TL=pgui.position()[0]
    #y_TL=pgui.position()[1]
    normal_mode()
else:
    parse_glossary()
#pgui.alert("ctrl -> normal mode  |  esc -> quit  |  shift -> glossary parse mode\n\nPANIC FAILSAFE = MOVE MOUSE TO TOP LEFT")
#state = get_key_state()

#if state == 1:
#    normal_mode()

#if state == -1:
#    parse_glossary()


#pgui.moveRel(100,0)