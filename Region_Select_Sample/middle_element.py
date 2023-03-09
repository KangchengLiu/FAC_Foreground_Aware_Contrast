
'''

@Written by: Dr. Kangcheng Liu

'''


# Python3 code to demonstrate working of 
# Using loop
  
# initializing list
test_list = [2, 3, 5, 7, 8, 5, 3, 5, 9]
  
# printing original list
print("The original list is : " + str(test_list))
  
# initializing K 
K = 5
  
# computing strt, and end index 
strt_idx = (len(test_list) // 2) - (K // 2)
end_idx = (len(test_list) // 2) + (K // 2)
  
# using loop to get indices 
res = []
for idx in range(len(test_list)):
      
    # checking for elements in range
    if idx >= strt_idx and idx <= end_idx:
        res.append(test_list[idx])
  
# printing result 
print("Extracted elements list : " + str(res))