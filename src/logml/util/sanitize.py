
#
# Sanitize strings: Make them valid for file names, dataframe names, etc
#

sanitize_valid_chars = set('_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
sanitize_dict = {
    '+': '_plus_',
    '-': '_',
    '=': '_eq_',
    '<': '_less_than_',
    '>': '_more_than_',
    '/': '_slash_',
}


def sanitize_name(s):
    ''' Sanitize a string to be used as a variable or column name '''
    return ''.join(sanitize_char(c) for c in str(s))


def sanitize_char(c):
    ''' Sanitize a string by only allowing "valid" characters '''
    if c in sanitize_valid_chars:
        return c
    if c in sanitize_dict:
        return sanitize_dict[c]
    return '_'
