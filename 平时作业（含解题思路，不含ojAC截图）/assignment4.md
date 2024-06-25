# Assignment #4: 排序、栈、队列和树

Updated 0005 GMT+8 March 11, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

Learn about Time complexities, learn the basics of individual Data Structures, learn the basics of Algorithms, and practice Problems.

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS

Python编程环境：vscode

C/C++编程环境：



## 1. 题目

### 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/



思路：尝试书写类。主要学到的点在于在`__init__`函数下，写`self._items=[]`可以规定self的属性为一个列表，在下面的函数中只要引用self.items，就可以直接用列表的方法了。

python中的collections模块中就有deque这一数据结构。与列表相比，`deque` 在内存使用上更高效。无论是在队列头部还是尾部进行操作，都具有常数级的复杂度。显然比下面的简易队列好的多。



代码

```python
class deque:
    def __init__(self):
        self._items=[]
    def add(self,x):
        self._items.append(x)
    def del_from_head(self):
        self._items.pop(0)
    def del_from_tail(self):
        self._items.pop(-1)
output=[]
case=int(input())
for _ in range(case):
    n=int(input())
    d=deque()
    for i in range(n):
        a,b=map(int,input().split())
        if a==1:
            d.add(b)
        elif a==2:
            if b==0:
                d.del_from_head()
            elif b==1:
                d.del_from_tail()
    if d._items:
        output.append(' '.join(map(str,d._items)))
    else:
        output.append('NULL')
for o in output:
    print(o)

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240314221302430](/Users/luzhuoran/Desktop/数算/作业/作业4/image-20240314221302430-0425585.png)



### 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



思路：**从右向左**遍历给定字符串，若遇到数字，则压入栈。则**越左边的数字就越接近栈顶**。当遍历到一个运算符，根据后序表达式的定义，需要计算该运算符右边的两个数字之间经过此运算后的结果，然后**将结果重新塞回原位继续计算剩下的表达式**。该运算符右边的两个数字就恰好是栈顶的和次栈顶的两个数字！最后输出栈内唯一剩下的数字，就是最终的结果。

1.`eval()`函数可以运行其内部字符串包含的表达式，在下面代码中可以返回一个数字。

2.`{:.6f}.format(float)`可以保留6位小数，以符合题目输出格式要求。



代码

```python
import math
operators={'+','-','*','/'}
l=input().split()
stack=[]
n=len(l)
for i in range(-1,-n-1,-1):
    letter=l[i]
    if letter not in operators:#数字
        stack.append(letter)#压入栈
    else:
        num1=stack.pop(-1)
        num2=stack.pop(-1)
        newnum=eval(str(num1)+letter+str(num2))
        stack.append(newnum)
print('{:.6f}'.format(stack[0]))
        
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240316215724517](/Users/luzhuoran/Desktop/数算/作业/作业4/image-20240316215724517-0597447.png)



### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



思路：

首先我们来考虑一个括号内的转化。对于一个括号内的表达式，可以写为这种格式：**乘除 加减 乘除 加减 乘除**。其中，乘除可以包括连乘连除或先乘后除。注意：中序和后序的**数字相对位置**是不变的。

对于**连续同级运算**的转换，例如1+2-3转化为12+3-，其规律为：从左向右遍历表达式，若遇到运算符，则证明前一个运算符已经算完了，所以在后序中添加前一个运算符。例如遍历到减号，则说明1+2的运算已经结束，按照同级运算从左到右的原则，应该在后序表达式里写12+。然后将减号储存起来。遍历完3后，储存器内剩了一个减号，把这个减号加到表达式最后。

接着考虑如加粗字体所示的表达式形式（没有括号），倘若遍历到一个加号或减号，就说明**其前面的乘除表达式已经计算完毕了**，根据上一段的内容，应该将储存器内剩余的一个乘除号添加到末尾。而后，倘若储存器中还剩余加减运算（当前加减号之前的加减号），那么说明**这个加减运算已经运算完毕**（因为这个加减运算的两个因子都计算完了），所以要把这个加减号写到表达式的末尾。最后，将当前加减号加入储存器。就可以一直运行下去。

```python
#一个括号内的转化
'''乘除加减乘除加减乘除->乘除乘除+-乘除+-'''
infix_notation=input()
l=infix_notation.split()
operators={'+','-','*','/'}
result_stack=[]#用于储存后序表达式的栈
operator_stack=[]#用于储存操作符的栈
n=len(l)
for i in range(n):
    letter=l[i]
    if letter not in operators:#letter是数字
        result_stack.append(letter)
    else:
        if letter=="*" or letter=='/':
          #letter前面的一个乘除号已经完成
            while (operator_stack[-1]=='*' or operator_stack[-1]=='/'):
                p=operator_stack.pop(-1)
                result_stack.append(p)
            operator_stack.append(letter)
        else:#加减的情况，letter前面的乘除和加减都完成了，这里加减号更靠近栈底，所以没问题
            while operator_stack:
                p=operator_stack.pop(-1)
                result_stack.append(p)
            operator_stack.append(letter)
while operator_stack:#最后把剩下的加减乘除号加到末尾
    p=operator_stack.pop(-1)
    result_stack.append(p)
print((result_stack))



```

如果**带括号**，那么带括号的部分要做到**单独运算**，括号内部的规则与上面是一样的。因此，可以界定一个“边界”，也就是左括号的位置。如果遇到了右括号，就需要把括号内剩余的加减乘除完成。相当于在每一个括号内运行了上面的代码。具体实现如下，函数`trans_input()`是将输入数据改成易于操作的列表形式。

代码

```python
'''带括号的部分独立运算
（表达式）->后序表达式，再与其他组合在一起
遇到(入栈，遇到操作符同上操作，遇到）弹出（之后所有操作符，相当于独立处理了这部分。
'''
def trans_input(string):
    '''处理输入数据,返回所需列表l'''
    operators={'+','-','*','/','(',')'}
    l=[]
    i=0
    string=string.strip()
    length=len(string)
    flag=False
    while i<length:
        letter=string[i]
        if letter!='.' and letter not in operators:
            if not flag:
                flag=True
                l.append(letter)
                i+=1
            else:
                l[-1]=l[-1]+letter
                i+=1
        elif letter in operators:
            flag=False
            l.append(letter)
            i+=1            
        elif letter=='.':
            temp='.'
            j=i+1
            while j<length and string[j].isdigit()==True:
                temp=temp+string[j]
                j+=1
            i=j
            l[-1]=l[-1]+temp
    return l
case=int(input())
output=[]
for _ in range(case):
    infix_notation=input()
    l=trans_input(infix_notation)
    operators={'+','-','*','/','(',')'}
    result_stack=[]
    operator_stack=[]
    n=len(l)
    for i in range(n):
        letter=l[i]
        if letter not in operators:
            result_stack.append(letter)
        else:
            if letter=='(':
                operator_stack.append(letter)
            elif letter=="*" or letter=='/':
                while operator_stack and operator_stack[-1]!='(' and (operator_stack[-1]=='*' or operator_stack[-1]=='/'):
                    p=operator_stack.pop(-1)
                    result_stack.append(p)
                operator_stack.append(letter)
            elif letter==')':#将括号内剩余的加减写出来
                while operator_stack[-1]!='(':
                    p=operator_stack.pop(-1)
                    result_stack.append(p)
                operator_stack.pop(-1)#删除'('
            else:#letter是加减的情况
                while operator_stack and operator_stack[-1]!='(':
                    p=operator_stack.pop(-1)
                    result_stack.append(p)
                operator_stack.append(letter)
    while operator_stack:#整体剩余的加减写出来
        p=operator_stack.pop(-1)
        result_stack.append(p)
    output.append(' '.join(result_stack))
for o in output:
    print(o)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240317175722835](/Users/luzhuoran/Desktop/数算/作业/作业4/image-20240317175722835-0669446.png)



### 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/



思路：  

**利用stack的FILO的性质**

入栈的顺序是降序排列（如6,5,4,3,2,1)，由于栈是FILO，那么出栈序列任意数A的后面比A大的数都是按照升序排列的；

入栈的顺序是升序排列（如1,2,3,4,5,6)，由于栈是FILO，那么出栈序列任意数A的后面比A小的数都是按照降序排列的。

以第二种情况为例，因为在出栈序列任意数A的后面比A小的数，都具有的特点是，比A早进栈而且比A晚出栈。那么这些数组成的序列就必然是恰好倒序的。

代码

```python
#23 生科 卢卓然
#将x中每个字符正序编号1~n
x=input()
L=len(x)
dic=dict()
i=1
for string in x:
    dic[string]=i
    i+=1
#提前声明一个maximum，防止oj特有的一种CE
maximum=-1

def check(index,length):#index当前字符的位置,length是s的长度
    '''鉴定该位置之后所有编号比他小的字符是否全为降序排列'''
    
    global s,maximum#maximum该位置之前字符中出现的最大编号
    number=dic[s[index]]
    if number<=maximum:
        return True
    '''之前的最大的编号的字符之后的所有编号比他小的字符全为降序排列,
    所以编号比maximum小的字符之后的就更是降序排列,剪枝'''
    
    tempmax=number#tempmax暂时的最大符号，不断更新以判断是否为降序排列
    flag=True#标记是否为降序排列
    for k in range(index+1,length):
        tempstr=s[k]
        tempnum=dic[tempstr]
        if tempnum<=number:#编号比number小
            if tempnum<=tempmax:#是否为降序排列
                tempmax=tempnum#更新tempmax
                continue
            else:
                flag=False
                break
    if flag:
        maximum=number#更新maximum
        return True
    else:
        return False

output=[]
while True:
    try:
        s = input()
        if set(s)==set(x) and len(set(s))==len(s):#防止蛇皮数据导致RE
            f=True#YES还是NO
            maximum=-1#初始化为一个比较小的值
            for j in range(L-2):#L-2:最后两位不用看，没有意义
                if not check(j,L):
                    f=False
                    break
            if f:
                output.append('YES')
            else:
                output.append("NO")
        else:
            output.append("NO")
    except EOFError:
        break

for o in output:
    print(o)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240317203425689](/Users/luzhuoran/Desktop/数算/作业/作业4/image-20240317203425689-0678866.png)



### 06646: 二叉树的深度

http://cs101.openjudge.cn/practice/06646/



思路：建树方式我选择了列表，搜索方式选择了bfs。需要注意的是题干中告诉我根节点就是1。往后就是bfs标准流程。每一次while q都对应了bfs的“一层”。对于每一层都要判断一下这一层有没有节点，也就是代码中的flag。



代码

```python
#建树
n=int(input())
l=[0]*(n+1)
for i in range(1,n+1):
    a,b=map(int,input().split())
    l[i]=(a,b)
#bfs
from collections import deque
visit=[False]*(n+1)
#根节点为1
q=deque()
q.append(1)
visit[1]=True
distance=1
while q:
    size=len(q)
    flag=False
    for _ in range(size):
        node=q.popleft()
        A,B=l[node]
        if (not visit[A]) and A!=-1:
            flag=True
            q.append(A)
            visit[A]=True
        if (not visit[B]) and B!=-1:
            flag=True
            q.append(B)
            visit[B]=True
    if flag:
        distance+=1
print(distance)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240317235806306](/Users/luzhuoran/Desktop/数算/作业/作业4/image-20240317235806306-0691087-0691088.png)



### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



思路：

Ultra-QuickSort题目使用的排序方法是归并排序（本题解答是基于归并排序的解法，用其他解法如冒泡排序时间复杂度较高）。在归并排序中，将序列递归地分为左右两半，分别排序后再合并到一起。在归并排序中，**合并函数的书写**是重点，也是本题关注的点。在两个有序序列进行合并的过程中，若通过**交换两个相邻数字的位置**来实现合并，一共需要交换多少次，就是本题需要解决的问题。

以下是归并排序（mergesort）的代码。`merge(arr,l,m,r)`函数是合并两个有序序列的函数。函数的主体是三个while。第一个while运用双指针法，是对两个序列的合并，涉及到了元素位置的改变。第二、三个while是简单地将L1或L2中的剩余有序元素复制到arr队尾，并不涉及到元素位置的改变。所以只需关注第一个while的内容。

```python
#归并排序
import sys
sys.setrecursionlimit(100000)
def merge(arr,l,m,r):
    '''对l到m和m到r两段进行合并'''
    n1=m-l+1#L1长
    n2=r-m#L2长
    L1=arr[l:m+1]
    L2=arr[m+1:r+1]
    ''' L1和L2均为有序序列'''
    i,j,k=0,0,l#i为L1指针，j为L2指针，k为arr指针
    '''双指针法合并序列'''
    while i<n1 and j<n2:
        if L1[i]<=L2[j]:
            arr[k]=L1[i]
            i+=1
        else:
            arr[k]=L2[j]
            j+=1
        k+=1
    while i<n1:
        arr[k]=L1[i]
        i+=1
        k+=1
    while j<n2:
        arr[k]=L2[j]
        j+=1
        k+=1
def mergesort(arr,l,r):
    '''对arr的l到r一段进行排序'''
    if l<r:#递归结束条件，很重要
        m=(l+r)//2
        mergesort(arr,l,m)
        mergesort(arr,m+1,r)
        merge(arr,l,m,r)
n=int(input())#序列长
array=[int(y) for y in input().split()]
mergesort(array,0,n-1)
print(array)
```

现在来分析用双指针法合并两个有序序列时，与其（复杂度上）等效的交换相邻数字的方法是怎样实现的。在双指针法中，是将 `L1[i]`和`L2[j]`中更小的一个放在arr的k处，在两个指针逐渐变大的过程中，将合并后的递增序列覆盖在`arr`的一段上。这种方法相对交换相邻数字，无疑很节省时间复杂度。交换相邻数字，关心的只是数字之间的相对位置，而不是绝对位置。所以可以假定这样的规则：当`L1[i]<=L2[j]`时，不改变数字的相对位置；当`L1[i]>L2[j]`时，将L2[j]通过不断换位向前移动，从而“插入”到L1[i]的前面一个位置。

那么L2[j]需要换多少次才能达到L1[i]前面呢？想象用交换位置法得到新序列的过程，那么在L2[j]动身之前，**L2中所有L2[j]之前的元素已经全部跑到了L1[i]的左边，并且与L1中L1[i]左边的元素组成了递增序列**。L2[j]的“目的地”就是这个已组成的递增序列和L1[i]之间的位置。L2和目的地之间相隔的元素，也就是**L1中L1[i]及其之后的元素**，其数量为`n1-i`（n1为原L1的长度）。所以在这一步中，交换的次数为`d+=(n1-i)`。

随着递归的进行，每一次合并中d不断累加，最终就可以得到总交换次数。

代码

```python
import sys
sys.setrecursionlimit(100000)
d=0
def merge(arr,l,m,r):
    '''对l到m和m到r两段进行合并'''
    global d
    n1=m-l+1#L1长
    n2=r-m#L2长
    L1=arr[l:m+1]
    L2=arr[m+1:r+1]
    ''' L1和L2均为有序序列'''
    i,j,k=0,0,l#i为L1指针，j为L2指针，k为arr指针
    '''双指针法合并序列'''
    while i<n1 and j<n2:
        if L1[i]<=L2[j]:
            arr[k]=L1[i]
            i+=1
        else:
            arr[k]=L2[j]
            d+=(n1-i)#精髓所在
            j+=1
        k+=1
    while i<n1:
        arr[k]=L1[i]
        i+=1
        k+=1
    while j<n2:
        arr[k]=L2[j]
        j+=1
        k+=1
def mergesort(arr,l,r):
    '''对arr的l到r一段进行排序'''
    if l<r:#递归结束条件，很重要
        m=(l+r)//2
        mergesort(arr,l,m)
        mergesort(arr,m+1,r)
        merge(arr,l,m,r)
results=[]
while True:
    n=int(input())#序列长
    if n==0:
        break
    array=[]
    for b in range(n):
        array.append(int(input()))
    d=0
    mergesort(array,0,n-1)
    results.append(d)
for r in results:
    print(r)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240318164342706](/Users/luzhuoran/Desktop/数算/作业/作业4/image-20240318164342706-0751426.png)



## 2. 学习总结和收获



这次作业的难度，让我仿佛又回到了上学期计概下半学期。但是数算相比于计概，更侧重了解更多数据结构的实现和应用、经典算法等等。这次的题目让我对栈、归并排序有了更加深刻的理解，也积累了一些经典算法。下周准备系统学习树相关题目，另外计划将各种数据结构相关经典应用和算法，在期末之前整理出来（当然也需要参考大佬们的）。加油！

ps：老师老师，请问我能进题解嘛，写了不少自己对于算法或题解代码的想法。**24591: 中序表达式转后序表达式**，**22068: 合法出栈序列**和**02299: Ultra-QuickSort**。谢谢老师！



