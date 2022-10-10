import re
import numpy as np


def rm_link(text):
    """
        Remove link, https prefixes
    """
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def rm_punct2(text):
    """
        Handle case like "shut up okay?Im only 10 years old"
        become "shut up okay Im only 10 years old"
    """
    return re.sub(
        r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)


def rm_html(text):
    """
        Remove html tags
    """
    return re.sub(r'<[^>]+>', '', text)


def space_bt_punct(text):
    """
        Add whitespaces between punctuation
        Remove double whitespaces
    """
    pattern = r'([.,!?-])'
    s = re.sub(pattern, r' \1 ', text)
    s = re.sub(r'\s{2,}', ' ', s)
    return s


def rm_number(text):
    """
        Remove number
    """
    return re.sub(r'\d+', '', text)


def rm_whitespaces(text):
    """
        Remove whitespaces
    """
    return re.sub(r' +', ' ', text)


def rm_nonascii(text):
    """
        Remove non-ascii characters
    """
    return re.sub(r'[^\x00-\x7f]', r'', text)


def rm_emoji(text):
    """
        Remove emoji
    """
    emojis = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE
    )
    return emojis.sub(r'', text)


def spell_correction(text):
    """
        Spell correction
    """
    return re.sub(r'(.)\1+', r'\1\1', text)


def clean_pipeline(text):
    """
        combine all preprocessing function
    """
    no_link = rm_link(text)
    no_html = rm_html(no_link)
    space_punct = space_bt_punct(no_html)
    no_punct = rm_punct2(space_punct)
    no_number = rm_number(no_punct)
    no_whitespaces = rm_whitespaces(no_number)
    no_nonasci = rm_nonascii(no_whitespaces)
    no_emoji = rm_emoji(no_nonasci)
    spell_corrected = spell_correction(no_emoji)
    return spell_corrected


def pad_features(sequences, pad_id, seq_length=128):
    """
        Padding string
    """
    features = np.full((len(sequences), seq_length), pad_id, dtype=int)

    for i, row in enumerate(sequences):
        # if seq_length < len(row) then review will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]

    return features