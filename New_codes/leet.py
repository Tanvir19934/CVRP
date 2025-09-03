def lastRemaining(n: int) -> int:
    if n==1:
        return 1
    arr = list(range(1, n+1))
    rev = False
    while True:
        new_arr = []
        if not rev:
            for i in range(0,len(arr)):
                if i%2!=0:
                    new_arr.append(arr[i])
        else:
            for i in range(len(arr)-2, -1, -2):
                new_arr.insert(0,arr[i])
        if len(new_arr)==1:
            return new_arr[0]
        arr = new_arr.copy()
        rev = not rev     
print(lastRemaining(6))