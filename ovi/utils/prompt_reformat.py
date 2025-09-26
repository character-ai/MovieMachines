import re

def format_prompt(text: str) -> str:
    """
    Convert user text into the model's expected format:
      * Replace case-insensitive `Audio:` (allowing optional space before colon)
        with <AUDCAP>…<ENDAUDCAP> up to end-of-line.
      * Replace quoted text (single or double, including curly forms)
        with <S>…<E>, never nesting, removing empty quotes.
      * Apostrophes inside words do NOT toggle quotes.
    """

    audio_pattern = r'(?i)\b(?:audio|aduio|audoi|adio|auido|audo|audioo)\s*:\s*(.*?)(?=$|\n)'
    def audio_replacer(m):
        return f"<AUDCAP>{m.group(1).rstrip()}<ENDAUDCAP>"
    text = re.sub(audio_pattern, audio_replacer, text)

    # --- 3. Quote handling ---
    quotes = {'"', "'"}
    result = []
    buffer = []
    inside_quote = False
    open_ch = None
    skip_literal_index = -1  # partner index for short elisions like 'n'

    i, n = 0, len(text)

    def flush_quote():
        """Emit buffer as <S>…<E>, keeping any trailing , or . inside."""
        nonlocal buffer, result
        if not buffer:
            return
        seg = ''.join(buffer).strip()
        if seg:
            result.append(f"<S>{seg}<E>")
        buffer = []

    while i < n:
        # --- Handle already-converted <AUDCAP> blocks atomically ---
        if text.startswith("<AUDCAP>", i):
            end = text.find("<ENDAUDCAP>", i)
            if end != -1:
                end += len("<ENDAUDCAP>")
                if inside_quote:
                    flush_quote()
                    inside_quote = False
                    open_ch = None
                result.append(text[i:end])
                i = end
                continue

        ch = text[i]

        # Escaped quote -> keep literal, drop backslash
        if ch == "\\" and i + 1 < n and text[i + 1] in quotes:
            (buffer if inside_quote else result).append(text[i + 1])
            i += 2
            continue

        # Word-internal apostrophe (e.g., can't) never toggles
        if ch == "'" and i > 0 and i + 1 < n and text[i-1].isalnum() and text[i+1].isalnum():
            (buffer if inside_quote else result).append(ch)
            i += 1
            continue

        if ch in quotes:
            if not inside_quote:
                # Remove empty quotes: quote + optional spaces + same quote
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j] == ch:
                    i = j + 1
                    continue
                inside_quote = True
                open_ch = ch
                buffer = []
                i += 1
                continue
            else:
                if i == skip_literal_index:
                    buffer.append(ch)
                    skip_literal_index = -1
                    i += 1
                    continue

                if ch == open_ch:
                    # Short elision pair like 'n'
                    if i + 2 < n and text[i + 1].isalpha() and text[i + 2] == open_ch:
                        buffer.append(ch)
                        skip_literal_index = i + 2
                        i += 1
                        continue
                    # Apostrophe starting alnum token after quote (e.g., 'em)
                    if i + 1 < n and text[i + 1].isalnum():
                        buffer.append(ch)
                        i += 1
                        continue
                    # Word-final elision like stayin'
                    if i - 2 >= 0 and text[i-2:i].lower() == "in":
                        buffer.append(ch)
                        i += 1
                        continue

                    # Close quote: keep any trailing , or . inside <E>
                    flush_quote()
                    inside_quote = False
                    open_ch = None
                    i += 1
                    continue
                else:
                    buffer.append(ch)
                    i += 1
                    continue

        # Normal char
        (buffer if inside_quote else result).append(ch)
        i += 1

    # Close if string ended inside a quote
    if inside_quote:
        flush_quote()

    return ''.join(result)