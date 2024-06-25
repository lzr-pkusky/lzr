# Assignment #6: "树"算：Huffman,BinHeap,BST,AVL,DisjointSet

Updated 2214 GMT+8 March 24, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）这次作业内容不简单，耗时长的话直接参考题解。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS 

Python编程环境：vscode

C/C++编程环境：



## 1. 题目

### 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/



思路：二叉搜索树依赖于这样一个性质：小于父节点的键都在左子树中，大于父节点的键则都在右子树中。我们称这个性质为二叉搜索性。数组第一个元素是根节点，紧跟着是小于根节点值的节点，在根节点左侧，直至遇到大于根节点值的节点，后续节点都在根节点右侧，按照这个思路递归即可。



代码

```python
class Node():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def buildTree(preorder):
    if len(preorder) == 0:
        return None

    node = Node(preorder[0])

    idx = len(preorder)
    for i in range(1, len(preorder)):
        if preorder[i] > preorder[0]:
            idx = i
            break
    node.left = buildTree(preorder[1:idx])
    node.right = buildTree(preorder[idx:])

    return node


def postorder(node):
    if node is None:
        return []
    output = []
    output.extend(postorder(node.left))
    output.extend(postorder(node.right))
    output.append(str(node.val))

    return output


n = int(input())
preorder = list(map(int, input().split()))
print(' '.join(postorder(buildTree(preorder))))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240402145424668](/Users/luzhuoran/Desktop/数算/作业/作业6/image-20240402145424668-2040865-2040866.png)



### 05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/



思路：

1. 使用`build_bst_from_list`函数，根据给定的无序列表构建二叉搜索树（BST）。
2. 利用队列（`deque`）来进行层次遍历（BFS），确保节点按顺序访问。
3. 遍历时，将当前节点的值加入输出列表，并将其子节点加入队列继续遍历。
4. 遍历完成后，将输出列表中的值转换为字符串并打印出来。

这样，我们就能得到BST的层次遍历结果，即从上到下、从左到右的节点值序列。

小技巧：`dict.fromkeys(numbers)` 创建一个字典，其中`numbers`列表中的元素作为键，值默认为`None`。由于字典的键是唯一的，这个步骤会移除列表中的重复元素。

代码

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert_into_bst(root, value):
    if root is None:
        return TreeNode(value)
    if value < root.value:
        root.left = insert_into_bst(root.left, value)
    else:
        root.right = insert_into_bst(root.right, value)
    return root

def build_bst_from_list(values):
    '''根据无序列表构建BST'''
    root = None
    for value in values:
        root = insert_into_bst(root, value)
    return root

l=map(int,input().split())
l=list(dict.fromkeys(l))#去除重复的数字
bst_root=build_bst_from_list(l)
#层次遍历/bfs
output=[]
from collections import deque
q=deque()#借助双端队列实现bfs
q.append(bst_root)
while q:
    size=len(q)
    for i in range(size):
        tmproot=q.popleft()
        output.append(tmproot.value)
        if tmproot.left:
            q.append(tmproot.left)
        if tmproot.right:
            q.append(tmproot.right)
print(' '.join(map(str,output)))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240402153958127](/Users/luzhuoran/Desktop/数算/作业/作业6/image-20240402153958127-2043599-2043600.png)



### 04078: 实现堆结构

http://cs101.openjudge.cn/practice/04078/

练习自己写个BinHeap。当然机考时候，如果遇到这样题目，直接import heapq。手搓栈、队列、堆、AVL等，考试前需要搓个遍。



思路：

完全二叉树的另一个有趣之处在于，可以用一个列表来表示它，而不需要采用“列表之列表”或“节点与引用”表示法。由于树是完全的，因此对于在列表中处于位置 p 的节点来说，它的左子节点正好处于位置 2p；同理，右子节点处于位置 2p+1。若要找到树中任意节点的父节点，只需使用 Python 的整数除法即可。给定列表中位置 n 处的节点，其父节点的位置就是 n//2。列表 heapList 的第一个元素是 0，它的唯一用途是为了使后续的方法可以使用整数除法。**（引自tree_questions）**

代码

```python
class Binheap:
    def __init__(self):
        self.heaplist=[0]
        self.currentSize=0
    def percup(self,i):
        '''将第i个元素向上推到应该在的位置'''
        while i//2>0:
            if self.heaplist[i]<self.heaplist[i//2]:
                tmp=self.heaplist[i//2]
                self.heaplist[i//2]=self.heaplist[i]
                self.heaplist[i]=tmp
            i//=2
    def insert(self,k):
        '''向二叉堆中添加一个元素并整理至有序'''
        self.heaplist.append(k)
        self.currentSize+=1
        self.percup(self.currentSize)
    def percdown(self,i):
        '''将第i个元素向下推到应该在的位置'''
        while i*2<=self.currentSize:
            mc=self.minchild(i)
            if self.heaplist[i]>self.heaplist[mc]:
                tmp=self.heaplist[i]
                self.heaplist[i]=self.heaplist[mc]
                self.heaplist[mc]=tmp
            i=mc
    def minchild(self,i):
        '''非叶子节点的第i个元素的两个子节点中更小的那个的索引'''
        if i*2+1>self.currentSize:
            return i*2
        else:
            if self.heaplist[i*2]<self.heaplist[i*2+1]:
                return i*2
            else:
                return i*2+1
    def delmin(self):
        '''取出根节点并重新整理二叉堆'''
        retval=self.heaplist[1]#二叉堆根节点、最小元素
        self.heaplist[1]=self.heaplist[self.currentSize]
        self.currentSize-=1
        self.heaplist.pop()
        self.percdown(1)
        return retval
alist=Binheap()
n=int(input())
output=[]
for i in range(n):
    numlist=[int(x) for x in input().split()]
    if len(numlist)==1:
        output.append(alist.delmin())
    else:
        number=numlist[1]
        alist.insert(number)
for o in output:
    print(o)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402142729621](/Users/luzhuoran/Desktop/数算/作业/作业6/image-20240402142729621.png)



### 22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/



思路：

这代码，真长啊！关于哈夫曼编码树的内容不再赘述，补充一些本题独有的小细节。

1、节点类Node的char属性应该改为一个集合，即`char=set()`，（作为给定默认值的参数应该放在所有参数的后面）。所以合并节点时，集合并集`left.char|right.char`。判断是否达到叶子节点：`len(node.char)==1`。若达叶子节点，需要把字典里的单个字母转化为字符串格式，可以用`''.join(node.char)`。

2、比较大小的规则：当频率相等时，可通过`min(self.char)<min(other.char)`比较。

代码

```python
import heapq
from collections import defaultdict
#创建树节点类
class Node:
    def __init__(self,freq,char=set()):
        self.char=char#节点对应的字符
        self.freq=freq#节点字符出现的频率
        self.left=None
        self.right=None
    def __lt__(self,other):
        #重载小于操作符，用于节点比较
        if self.freq<other.freq:
            return True
        elif self.freq>other.freq:
            return False
        else:
            return min(self.char)<min(other.char)
#创建优先队列并建立哈夫曼树
def create_tree(frequencies):
    #使用频率创建节点，并加入优先队列
    pq=[Node(freq,set(char)) for char,freq in frequencies.items()]
    heapq.heapify(pq)
    while len(pq)>1:
        #循环只到堆中只剩一个节点
        left=heapq.heappop(pq)
        right=heapq.heappop(pq)
        merged=Node(left.freq+right.freq,left.char|right.char)#合并这两个节点
        merged.left=left
        merged.right=right
        heapq.heappush(pq,merged)#将合并后的节点加入堆中
    return pq[0]
#霍夫曼编码
def huffman_encoding(node,prefix='',code={}):
    if node != None:
        if len(node.char)==1:
            code[''.join(node.char)]=prefix
        huffman_encoding(node.left,prefix+'0',code)
        huffman_encoding(node.right,prefix+'1',code)
    return code

def count_char_frequency(text):
    frequency=defaultdict(int)
    for char in text:
        frequency[char]+=1
    return frequency
#编码文本
def encode(text):
    frequencies=count_char_frequency(text)
    root=create_tree(frequencies)
    huffman_code=huffman_encoding(root)
    encoded_text=''.join([huffman_code[char] for char in text])
    return encoded_text,huffman_code
#解码文本
def decode(encoded_text,huffman_code):
    reverse_code={v:k for k,v in huffman_code.items()}
    code=''
    decoded_text=''
    for digit in encoded_text:
        code+=digit
        if code in reverse_code:
            decoded_text+=reverse_code[code]
            code=''
    return decoded_text

n=int(input())
text=''
for i in range(n):
    char,num=input().split()
    num=int(num)
    text+=char*num
encoded_text,huffman_code=encode(text)
output=[]
while True:
    try:
        s=input()
        if s.isdigit():
            output.append(decode(s,huffman_code))
        else:
            encodedresult=''
            for string in s:
                encodedresult+=huffman_code[string]
            output.append(encodedresult)
    except EOFError:
        break
for o in output:
    print(o)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402204328077](/Users/luzhuoran/Desktop/数算/作业/作业6/image-20240402204328077-2061809-2061810.png)



### 晴问9.5: 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359



思路：AVL树的建立。代码中看似没有从下向上检查祖先节点是否不平衡的代码部分，实际上这藏在了`_insert()`函数的递归当中，通过递归自底向上地检查祖先节点是否失衡！这个很重要。



代码

```python
class Node:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
        self.height=1#树的高度
class AVL:
    '''二叉平衡搜索树'''
    def __init__(self):
        self.root=None#self.root表示AVL树根节点
    def insert(self,value):
        '''向avl树中插入值为value的元素'''
        if not self.root:#空树则创建树
            self.root=Node(value)
        else:
            self.root=self._insert(value,self.root)#非空则插入
    def _insert(self,value,node):#value为新插入的点的值
      '_insert()函数的递归当中自底向上地检查祖先节点是否失衡'
        if not node:#空树则建树
            return Node(value)
        elif value<node.value:#应归于左子树
            node.left=self._insert(value,node.left)
        else:#应归于右子树
            node.right=self._insert(value, node.right)
				node.height=1+max(self._get_height(node.left),
                      self._get_height(node.right))
        #二叉树高度的递推式
        
        #调整平衡
        balance = self._get_balance(node)#当前节点的平衡因子
        
        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        return node
    
    def _get_height(self, node):
        '''或许node的树高度'''
        if not node:
            return 0
        return node.height
    def _get_balance(self, node):
        '''获取节点node的平衡因子'''
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z):
        '''左旋'''
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        '''右旋'''
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    
    def preorder(self):
        '''输出前序遍历'''
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402223742749](/Users/luzhuoran/Desktop/数算/作业/作业6/image-20240402223742749-2068663.png)



### 02524: 宗教信仰

http://cs101.openjudge.cn/practice/02524/



思路：将一个集合的根节点指向另一个集合的根节点，从而合并两个集合！都是根节点和根节点之间的操作！路径压缩，直接将x的根节点设置为其祖先的根节点！



代码

```python
class UnionFind:
    # 初始化并查集，每个元素的根节点是它自己
    def __init__(self, size):
        self.root = [i for i in range(size)]

    # 查找元素x的根节点
    def find(self, x):
        if x == self.root[x]:
            return x
        # 路径压缩，直接将x的根节点设置为其祖先的根节点
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    # 合并元素x和y所在的集合
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            # 将一个集合的根节点指向另一个集合的根节点，从而合并两个集合
            self.root[rootY] = rootX

    # 计算并查集中不同集合的数量
    def count(self):
        # 通过查找每个元素的根节点，并去重，得到不同集合的数量
        return len(set(self.find(i) for i in range(len(self.root))))
output=[]
flag=0
while True:
        flag+=1
        # 使用并查集来计算宗教数目的上限
        n, m = map(int, input().split())
        if n==m==0:
            break
        uf = UnionFind(n)  # 创建一个大小为n的并查集
        for i in range(1, m + 1):
            # 解析每一行的两个学生编号，并将它们所在的集合合并
            x, y = map(int, input().split())
            uf.union(x - 1, y - 1)  # 学生编号从1开始，因此需要减1
        output.append(uf.count())  # 返回不同宗教的数目上限

for y in range(1,flag):
    print(f'Case {y}: {output[y-1]}')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402234401618](/Users/luzhuoran/Desktop/数算/作业/作业6/image-20240402234401618-2072642.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

本次作业难度很大，但收获也很大。从下午一点学到晚上12点，学完了Huffman,BinHeap,BST,AVL,DisjointSet这五项内容的代码实现，完成了作业的例题，感觉收获很大，认知得到了升华。树这一体系已经逐渐趋于完善了，assignment P也在同步整理中。



