import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
vocab_size = encoding.max_token_value + 1
encode = lambda s: encoding.encode(s)
decode = lambda l: encoding.decode(l)