# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

2024 spring, Complied by ==卢卓然 生命科学学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS 

Python编程环境：vscode

C/C++编程环境：

## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：直接转化为列表，然后倒序后输出。



代码

```python
# 
print(' '.join(reversed(input().split())))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240409230618565](/Users/luzhuoran/Desktop/数算/作业/作业7/image-20240409230618565-2675180.png)



### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：

思路就是利用双端队列完成搜索，每搜索到一个word，lookups就加上一，最终统计。

代码

```python
from collections import deque

M, N = map(int, input().split())
words = list(map(int, input().split()))

memory = deque()
lookups = 0

for word in words:
    if word not in memory:
        if len(memory) == M:
            memory.popleft()
        memory.append(word)
        lookups += 1

print(lookups)
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240409225101678](/Users/luzhuoran/Desktop/数算/作业/作业7/image-20240409225101678.png)



### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：

1. 如果k等于0，那么如果数组a的第一个元素大于1，则x等于1，否则x等于-1。
2. 如果k等于n，那么x等于数组a的最后一个元素。
3. 其他情况下，检查第k个元素是否是唯一满足条件的。如果是，则x等于第k个元素；否则，x等于-1。



代码

```python
n, k = map(int, input().split())

a = list(map(int, input().split()))
a.sort()

# 寻找 x
if k == 0:
    x = 1 if a[0] > 1 else -1
elif k == n:
    x = a[-1]
else:
    # 检查第 k 个元素是否是唯一满足条件的
    x = a[k-1] if a[k-1] < a[k] else -1

print(x)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240409225451513](/Users/luzhuoran/Desktop/数算/作业/作业7/image-20240409225451513-2674495.png)



### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



思路：

建立fbi树就是遵循规则；后序遍历就是递归！

代码

```python
class Node:
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None

def build_FBI(string):
    root = Node()
    if '0' not in string:
        root.value = 'I'
    elif '1' not in string:
        root.value = 'B'
    else:
        root.value = 'F'
    l = len(string) // 2
    if l > 0:
        root.left = build_FBI(string[:l])
        root.right = build_FBI(string[l:])
    return root

def post_traverse(node):
    ans = []
    if node:
        ans.extend(post_traverse(node.left))
        ans.extend(post_traverse(node.right))
        ans.append(node.value)
    return ''.join(ans)

n = int(input())
string = input()
root = build_FBI(string)
print(post_traverse(root))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240409232806356](/Users/luzhuoran/Desktop/数算/作业/作业7/image-20240409232806356-2676487-2676489.png)



### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：

先初始化组和成员与组的映射关系。先假定一个成员的id代表组id。在处理数据的过程中，如果队列为空，那么要删除，这一点被我忽略过。

代码

```python
from collections import deque	

t = int(input())
groups = {}
member_to_group = {}



for _ in range(t):
    members = list(map(int, input().split()))
    group_id = members[0]  
    groups[group_id] = deque()
    for member in members:
        member_to_group[member] = group_id


queue = deque()
queue_set = set()


while True:
    command = input().split()
    if command[0] == 'STOP':
        break
    elif command[0] == 'ENQUEUE':
        x = int(command[1])
        group = member_to_group.get(x, None)
        # Create a new group if it's a new member not in the initial list
        if group is None:
            group = x
            groups[group] = deque([x])
            member_to_group[x] = group
        else:
            groups[group].append(x)
        if group not in queue_set:
            queue.append(group)
            queue_set.add(group)
    elif command[0] == 'DEQUEUE':
        if queue:
            group = queue[0]
            x = groups[group].popleft()
            print(x)
            if not groups[group]: 
                queue.popleft()
                queue_set.remove(group)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240409231957939](/Users/luzhuoran/Desktop/数算/作业/作业7/image-20240409231957939-2675999.png)



### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：1.通过字典建立输入数据的父子关系；2.找到树的根（这里我将父节点和子节点分别用两个列表记录，最后使用集合减法）；3.通过递归实现要求的从小到大遍历。



代码

```python
from collections import defaultdict
n = int(input())
tree = defaultdict(list)
parents = []
children = []
for i in range(n):
    t = list(map(int, input().split()))
    parents.append(t[0])
    if len(t) > 1:
        ch = t[1::]
        children.extend(ch)
        tree[t[0]].extend(ch)


def traversal(node):
    seq = sorted(tree[node] + [node])
    for x in seq:
        if x == node:
            print(node)
        else:
            traversal(x)


traversal((set(parents) - set(children)).pop())
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240409230154237](/Users/luzhuoran/Desktop/数算/作业/作业7/image-20240409230154237-2674915.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==



本次月考题目前几道较简单，后几道较难。临近期中，还是计划先把笔试题目看一看，补一补系统的知识体系。

