# can be usefull
# https://regex101.com/
# https://www.regular-expressions.info/

# funny game
# https://alf.nu/RegexGolf


import re

x = "aXaXXaXXXa"
print(x)

re.findall("a", x)
re.sub("a", "b", x)
re.split("a", x)

# alternatives
x = """Oh blimey, how time flies. Sadly we are reaching the end of yet 
       another programme and so it is finale time. We are proud to be 
       bringing to you one of the evergreen bucket kickers. Yes, the 
       wonderful death of the famous English Admiral Nelson."""

re.findall("m|f", x)
re.findall("[mf]", x)

re.findall("it|is", x)
re.findall("i[ts]", x) # or () for alterantive, but
re.findall("i(t|s)", x) # in that case findall returns captured groups
re.findall("i(?:t|s)", x) # (?:pattern) - non-capturing group
re.findall("(i(t|s))", x) # or catch also a group of full pattern

# special characters
# .\|()[{}ˆ$*+?

x = "What? Nothing."
re.findall("?", x)
re.findall(".", x)

re.findall("\\?", x)
re.findall("\\.", x)

# . - any character
# * – 0 or more times
# + – 1 or more times
# ? – 0 or 1 times
# {n,} – at least n times
# {,m} – at most m times
# {n,m} – at least n but no more than m times

# e. g.:
# ab* -> a, ab, abb, ...
# ab+ -> ab, abb, abbb, ...
# (ab)+ -> ab, abab, ...


x = "aaXXaXaXXaaXaaXaaaXaX"

re.findall("a+", x)
re.findall("aX+", x)
re.findall("(?:aX)+", x)
re.findall("a{2,3}X", x)

# quantifiers are greedy, put ? after to be non greedy 
# see: \\(.+\\) vs \\(.+?\\)
x = "napis (pierwszy nawias) cos innego (drugi nawias)"
re.findall("\\(.+\\)", x)
re.findall("\\(.+?\\)", x)


# anchors:
# ^ - matching at the beginning
# $ - matching at the end

x = ["NAPIS jest na początku", 
     "tutaj NAPIS jest w środku",
     "na końcu jest NAPIS",
     "NAPIS",
     "NAPIS na początku i na końcu NAPIS"]

[re.findall("^NAPIS", e) for e in x]
[re.findall("NAPIS$", e) for e in x]
[re.findall("^NAPIS$", e) for e in x]
[re.findall("^NAPIS.*NAPIS$", e) for e in x]

# basic character classes
# [0-9], \d - digit
# \D, [^\d] - nondigit
# [a-zA-Z0-9_], \w - word charcter
# \s - whitespace (" " ,"\t", "\n", "\r")

x = "Some 45_6 text\n with  whitespaces and 123 \t to find."
re.findall("\\s", x)
re.findall("\\d", x)
re.findall("\\D", x)
re.findall("\\w", x)
re.findall("\\w+", x)

# brackets () catch groups, we can use these groups later
re.sub('(\\d).(\\d)', '\\2-\\1', '123|456|789')

# 123: 
# (\\d) : 1, (1st group is referred to as  \\1)
# .     : 2,
# (\\d) : 3, (2nd group is referred to as  \\2)

