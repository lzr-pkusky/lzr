# Assignment #5: "树"算：概念、表示、解析、遍历

Updated 2124 GMT+8 March 17, 2024

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

Python编程环境：PyCharm 

C/C++编程环境：

## 1. 题目

### 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/



思路：求二叉树高度与之前是一样的，题目里隐晦的一点是要自己寻找根节点！！！也就是没有父亲节点的节点，也就是不是别人的子节点的节点。



代码

```python
#建树
n=int(input())
num=0
l=[0]*(n)
hasparents=[False]*n
for i in range(0,n):
    a,b=map(int,input().split())
    if a==b==-1:
        num+=1
    if a!=-1:
        hasparents[a]=True
    if b!=-1:
        hasparents[b]=True
    l[i]=(a,b)
root=hasparents.index(False)
#bfs
from collections import deque
visit=[False]*(n)
#根节点为n
q=deque()
q.append(root)
distance=0
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
print(str(distance)+' '+str(num))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240325170432810](/Users/luzhuoran/Desktop/数算/作业/作业5/image-20240325170432810-1357475.png)



### 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/



思路：学习了题解的思路，首先定义了treenode类型，允许一个节点有多个子节点。然后是处理字符串的函数。如果是字母，创建新节点，如果栈不空说明此节点是栈顶的子节点；遇到左括号，当前节点可能会有子节点，压入栈进行处理。遇到有括号子节点列表结束，弹出当前节点。这样就建立起了树，最后的node就对应了整个树。获得了树之后，再进行遍历。



代码

```python
class TreeNode:
    def __init__(self, value): #类似字典
        self.value = value
        self.children = []

def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点


def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)

# 主程序
def main():
    s = input().strip()
    s = ''.join(s.split())  # 去掉所有空白字符
    root = parse_tree(s)  # 解析整棵树
    if root:
        print(preorder(root))  # 输出前序遍历序列
        print(postorder(root))  # 输出后序遍历序列
    else:
        print("input tree string error!")

if __name__ == "__main__":
    main()
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240326211354682](/Users/luzhuoran/Desktop/数算/作业/作业5/image-20240326211354682-1458835-1458836.png)



### 02775: 文件结构“图”

http://cs101.openjudge.cn/practice/02775/



思路：仍然是学习题解。把目录看成节点，每个目录节点设置子目录节点列表和子文件列表。输出格式：当前目录，遍历子目录，遍历子文件。

发现数算题目的特点就是定义类、函数来简化或者说整合运算过程。



代码

```python
class Node:
    def __init__(self,name):
        self.name=name
        self.dirs=[]
        self.files=[]

def print_(root,m):
    pre='|     '*m
    print(pre+root.name)
    for Dir in root.dirs:
        print_(Dir,m+1)
    for file in sorted(root.files):
        print(pre+file)
        
tests,test=[],[]
while True:
    s=input()
    if s=='#':
        break
    elif s=='*':
        tests.append(test)
        test=[]
    else:
        test.append(s)
for n,test in enumerate(tests,1):
    root=Node('ROOT')
    stack=[root]
    print(f'DATA SET {n}:')
    for i in test:
        if i[0]=='d':
            Dir=Node(i)
            stack[-1].dirs.append(Dir)
            stack.append(Dir)
        elif i[0]=='f':
            stack[-1].files.append(i)
        else:
            stack.pop()
    print_(root,0)
    print()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240326214522890](/Users/luzhuoran/Desktop/数算/作业/作业5/image-20240326214522890-1460724.png)



### 25140: 根据后序表达式建立队列表达式

http://cs101.openjudge.cn/practice/25140/



思路：问题是要求将后缀表达式转换为等效的队列表达式。队列表达式是通过反转从后缀表达式构建的表达式树的级别顺序遍历来获得的。

以下是一个循序渐进的计划：
1.创建一个TreeNode类来表示树中的每个节点。2.创建一个函数build_tree，该函数以后缀表达式作为输入，并返回构建的树的根。使用堆栈来存储节点。遍历后缀表达式中的字符。如果字符是操作数，请创建一个新节点并将其推送到堆栈上。如果角色是运算符，则从堆栈中弹出两个节点，使它们成为新节点的子节点，并将新节点推送到堆栈上。3.创建一个以树的根为输入并返回树的级别顺序遍历的函数level_order_transversal。使用队列遍历来存储要访问的节点。当队列不为空时，将节点排成队列，访问它，并将其子节点排入队列。4.对于每个后缀表达式，构造树，执行级别顺序遍历，反转结果并输出。



代码

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(postfix):
    stack = []
    for char in postfix:
        node = TreeNode(char)
        if char.isupper():
            node.right = stack.pop()
            node.left = stack.pop()
        stack.append(node)
    return stack[0]

def level_order_traversal(root):
    queue = [root]
    traversal = []
    while queue:
        node = queue.pop(0)
        traversal.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return traversal

n = int(input().strip())
for _ in range(n):
    postfix = input().strip()
    root = build_tree(postfix)
    queue_expression = level_order_traversal(root)[::-1]
    print(''.join(queue_expression))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240326214755128](/Users/luzhuoran/Desktop/数算/作业/作业5/image-20240326214755128-1460876.png)



### 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/practice/24750/



思路：

学习题解思路。重点是要利用后序遍历中最后一个元素是当前根节点这个重要性质！

代码

```python
"""
定义一个递归函数。在这个递归函数中，我们将后序遍历的最后一个元素作为当前的根节点，然后在中序遍历序列中找到这个根节点的位置，
这个位置将中序遍历序列分为左子树和右子树。
"""
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None

    # 后序遍历的最后一个元素是当前的根节点
    root_val = postorder.pop()
    root = TreeNode(root_val)

    # 在中序遍历中找到根节点的位置
    root_index = inorder.index(root_val)

    # 构建右子树和左子树
    root.right = buildTree(inorder[root_index + 1:], postorder)
    root.left = buildTree(inorder[:root_index], postorder)

    return root


def preorderTraversal(root):
    result = []
    if root:
        result.append(root.val)
        result.extend(preorderTraversal(root.left))
        result.extend(preorderTraversal(root.right))
    return result


# 读取输入
inorder = input().strip()
postorder = input().strip()

# 构建树
root = buildTree(list(inorder), list(postorder))

# 输出前序遍历序列
print(''.join(preorderTraversal(root)))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240326214943307](/Users/luzhuoran/Desktop/数算/作业/作业5/image-20240326214943307-1460984.png)



### 22158: 根据二叉树前中序序列建树

http://cs101.openjudge.cn/practice/22158/



思路：和24750基本相同。只不过变成了利用前序序列的第一个元素是根节点。



代码

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root
def postorder_traversal(root):
    if root is None:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.value

while True:
    try:
        preorder = input().strip()
        inorder = input().strip()
        root = build_tree(preorder, inorder)
        print(postorder_traversal(root))
    except EOFError:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240326215309915](/Users/luzhuoran/Desktop/数算/作业/作业5/image-20240326215309915-1461190.png)



## 2. 学习总结和收获

本次作业涉及到了树的列表写法、队列写法、括号写法；还有三个遍历及其相互转化。关系是：已知中序和前/后序可以推出后/前序，但是已知前序和后序是不能推出中序的。





