try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources
    
    
import ast

def load_data(filename: str):

    with resources.open_text('resources', filename) as file:
        text = file.read()

    content = ast.literal_eval(text)
    return content




vowels_path = "start_vowels.txt"
start_vowels = load_data(vowels_path)

huyen = start_vowels['huyen']
sac = start_vowels['sac']
nang = start_vowels['nang']
hoi = start_vowels['hoi']
nga = start_vowels['nga']
khong_dau = start_vowels['khong_dau']

list_start_vowels = []
list_start_vowels.extend(huyen)
list_start_vowels.extend(sac)
list_start_vowels.extend(nang)
list_start_vowels.extend(hoi)
list_start_vowels.extend(nga)
list_start_vowels.extend(khong_dau)

rhyme_path = "rhymes.txt"

rhymes_dict = load_data(rhyme_path)


even_chars = []

even_chars.extend(huyen)
even_chars.extend(khong_dau)


co_dau = []
co_dau.extend(huyen)
co_dau.extend(sac)
co_dau.extend(nang)
co_dau.extend(hoi)
co_dau.extend(nga)

tone_dict = load_data("tone_dict.txt")




def split_word(word):
    """
        Split word by 2 part, starting and ending

        param word: word to split

        return: ending part of word
        Ex: mùa -> ùa
    """
    word_length = len(word)
    start_index = 0
    prev = ''
    for i in range(word_length):
        if prev == 'g' and word[i] == 'i':
            continue
        if prev == 'q' and word[i] == 'u':
            continue
        if word[i] in list_start_vowels:
            start_index = i
            break
        prev = word[i]
    return word[start_index:]



def compare(word1: str, word2: str):
    """
      Check 2 words rhyme if the same

      param word1, word2: words to check

      return: is the same rhyme or not
    """
    rhyme1 = split_word(word1)
    rhyme2 = split_word(word2)

    if rhyme2 in rhymes_dict[rhyme1]:
        return True
    return False



def get_tone(word: str):
    """
          Check word is even tone or not

          param word: word to check tone

          return: even or uneven
        """
    first_char = split_word(word)
    
    try:
        first_char, second_char, third_char = first_char[0], first_char[1], first_char[2]
            
        for i in co_dau:
            if i == first_char or i == second_char or i == third_char:
                if i in huyen:
                    return 'even', 'huyen'
                else:
                    return 'uneven', 'other'
        return 'even', 'ngang'
    
    except:
        
        try:
            first_char, second_char = first_char[0], first_char[1]
            
            for i in co_dau:
                if i == first_char or i == second_char:
                    if i in huyen:
                        return 'even', 'huyen'
                    else:
                        return 'uneven', 'other'
            return 'even', 'ngang'
            
        except:
            first_char = first_char[0]
            for i in even_chars:
                if first_char == i:
                    if first_char in huyen:
                        return 'even', 'huyen'
                    else:
                        return 'even', 'ngang'
                
            return 'uneven', 'other'
        
        
def check_tone_sentence(sentence: str | list):
    """
        Check sentence is on the right form of even or uneven rule

        param sentence: sentence to check tone

        return: sentences after added notation to highlight error
                total_wrong_tone: total wrong tone in sentence
      """
    if isinstance(sentence, list):
        words = sentence
    else:
        words = sentence.split(" ")
    length = len(words)
    
    if length == 8:
        cur_tone_dict = tone_dict[8]
    elif length == 6:
        cur_tone_dict = tone_dict[6]
    else:
        return ['Sai số từ'], 20
    
    wrong_tone_words = []
    wrong_tone_pair = None
    
    for i in cur_tone_dict:
        tone = get_tone(words[i])
        if tone[0] != cur_tone_dict[i]:
            wrong_tone_words.append(words[i])

        if i == 5:
            dau = tone[1]
            tu = words[i]
        elif i == 7:
            if (dau == 'huyen' and tone[1] == "ngang") or (dau == 'ngang' and tone[1] == 'huyen'):
                pass
            else:
                wrong_tone_pair = f"Cặp từ thứ 6 ('{tu}') và 8 ('{words[i]}') chưa thoả mãn đủ 1 thanh ngang và 1 thanh huyền"
        else:
            pass   

    return wrong_tone_words, wrong_tone_pair

# check_tone_sentence('tôi là lê hoàng anh đức hà tĩnh')



def check_luc_bat_rule(poem: str):
    """
    Kiểm tra đoạn thơ có tuân thủ đúng quy tắc thơ lục bát không:
    - Đúng chẵn dòng
    - Đúng số chữ (6-8)
    - Đúng vần (lục với bát, bát với lục tiếp theo)
    - Đúng thanh điệu (even/uneven theo tone_dict)
    """
    
    score = 0
    
    ## chia thành các dòng tổng số dòng phải chẵn
    lines = [l.strip() for l in poem.splitlines() if l.strip()]
    # print(f"----Tổng thể----")
    
    if len(lines) % 2 != 0:
        # print(-100 ,f"\\- Số câu {len(lines)} là lẻ: ")
        score += -100
        return score
    else:
        previous_bat_8th_word = None
        
        max_score_per_line = 100//(len(lines)*2)
        for i in range(0, len(lines), 2):
            # print(f"----Cặp lục bát thứ {i//2+1}----")
            luc = lines[i].split()
            bat = lines[i+1].split()
            
            
            if len(luc) == 6 and len(bat) == 8:
                # 2. Gieo vần
                luc_6th_word = luc[-1]
                bat_6th_word = bat[5]
                bat_8th_word = bat[-1]
                            
                if previous_bat_8th_word:
                    if compare(previous_bat_8th_word, luc_6th_word):
                        pass
                    else:
                        # print(-max_score_per_line, f"vần cuối ('{luc_6th_word}') của câu 6 ({lines[i]}) không khớp với vần cuối ({previous_bat_8th_word}) của câu 8 trước nó.")
                        score += -max_score_per_line
                        
                    
                if compare(luc_6th_word, bat_6th_word):
                    pass 
                else:
                    # print(-max_score_per_line, f"vần cuối ({luc_6th_word}) của câu 6 ({lines[i]}) không khớp với vần thứ 6 ({bat_6th_word}) của câu 8 sau nó {lines[i+1]}.")
                    score += -max_score_per_line
                                
                if bat_8th_word:
                    previous_bat_8th_word = bat_8th_word
                    
                # 3. Thanh điệu

                luc_tone_check = check_tone_sentence(luc)
                if luc_tone_check[0]:
                    # print(-max_score_per_line, f"các từ {luc_tone_check[0]} trong câu 6 ({lines[i]}) sai thanh điệu.")
                    score += -max_score_per_line/3*len(luc_tone_check[0])
                        
                    
                bat_tone_check = check_tone_sentence(bat)
                if bat_tone_check[0]:
                    # print(-max_score_per_line, f"các từ {bat_tone_check[0]} trong câu 6 ({lines[i+1]}) sai thanh điệu.")
                    score += -max_score_per_line/5*len(luc_tone_check[0])
                    
                if bat_tone_check[1]:
                    # print(-max_score_per_line/5, bat_tone_check[1])
                    score += -max_score_per_line/5
                        
            else:
                # print(-max_score_per_line * 4, f"Cặp Câu '{lines[i]}' và '{lines[i+1]}' sai số từ.")
                score += -max_score_per_line * 4
        return score
            
            
        

