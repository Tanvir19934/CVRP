def palindrome(x):
    if len(x)==0 or len(x)==1:
        return True
    elif x[0]==x[len(x)-1]:
        return palindrome(x[1:-1])
    else: return False

def string_reverse(x):
    if x=="":
        return ""
    return string_reverse(x[1:]) + x[0]

def dec_bin(x):
    rem = x%2
    if x//2==0:
        return str(rem)
    return dec_bin(x//2) + str(rem)

def sum_of_naturals(x):
    if x==0 or x==1:
        return x
    return x + sum_of_naturals(x-1)



print(palindrome("racecar"))
print(string_reverse("hello"))
print(dec_bin(238))
print(sum_of_naturals(10))

print(string_reverse("tanvir"))
