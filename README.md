# Quick Reference



## 3. Matrices



:pushpin: **Eigenvalues and Eigenvectors**

The eigenvalues of an $n\times n$ matrix $M$ are equal to the roots of the characteristic polynomial given by
$$
\text{det}(M-\lambda I)
$$
An eigenvector $V$ associated with the eigenvalue $λ$ of the matrix $M$ is given by the solution to the homogeneous linear system
$$
(M-λI)V= 0
$$
The eigenvalues of a real symmetric matrix are real, and the eigenvectors corresponding to distinct eigenvalues of a real symmetric matrix are orthogonal.



:pushpin: **Diagonalization**

If $V_1,V_2,...,V_n$ are linearly independent eigenvectors of an $n\times n$ matrix $M$, then the matrix $A$ given by
$$
A = \begin{bmatrix} V_1&V_2&...&V_n\end{bmatrix}
$$
diagonalizes $M$, meaning that 
$$
A^{−1}MA=
\begin{bmatrix}
\lambda_1&0&\cdots&0\\
0&\lambda_2&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&\lambda_n\\
\end{bmatrix}
$$
where $\lambda_1,\lambda_2,...,\lambda_n$ are the eigenvalues of $M$.




# Chapter 2 Vector

## 2.4 Vector Space

> ​	⭐ **Definition 2.10.** A vector space is a set $V$, whose elements are called vectors, for which addition and scalar multiplication are defined, and the following properties hold.

- $P\in V,\space Q\in V, \quad \text{s.t.  }P+Q\in V$
- $P\in V, a\in \mathbb{R}, \text{  s.t.  }aP\in V$
- $\exist\space 0\in V, \text{s.t.}\quad\ P+0=P$.
- $\exist\space Q\in V, \text{s.t.}\quad\ P+Q=0$.
- $(P+Q)+R=P+(Q+R)$
-  $(ab)P=a(bP)$
- $a(P+Q)=aP+aQ$
- $(a+b)P = aP+bP$







# Chapter 3 Matrices


## 3.2 Linear Systems

> ​	Matrix is very handy in solving **linear equations**:

$$
\begin{align}
3x+2y-3z&=-13\\
4x-3y+6z&=7\\
x-z&=-5
\end{align}
$$

> ​	can illustrated as **matrix format**:

$$
\begin{bmatrix}
3&2&-3\\
4&-3&6\\
1&0&-1
\end{bmatrix}
\begin{bmatrix}
x\\y\\z
\end{bmatrix}=
\begin{bmatrix}
-13&7&5
\end{bmatrix}
$$

> ​	looking in a **big picture**:

$$
Ax=b
$$

- $A$ , the **coefficient matrix**
- $x$, unknown
- $b$, right-hand side, **constant vector**





> ​	:pushpin: **nonhomogeneous** : right hand side is nonzero
>
> ​	:pushpin: **homogeneous** : right hand side is full of zero





> ​	⭐**Definition 3.3. Elementary row operation**: It is one of the following three operations that can be performed on a matrix

1. exchange 2 rows
2. multiply a row by a nonzero scalar
3. add a multiple of one row to another row





> ​	⭐ **Definition 3.4.** A matrix is in **reduced form** if and only if it satisfies the following conditions.

1. For every nonzero row, the leftmost nonzero entry, called the leading entry, is 1.
2. Every nonzero row precedes every row of zeros. That is, all rows of zeros reside at the bottom of the matrix.
3. If a row’s leading entry resides in column $j$, then no other row has a nonzero entry in column $j$.
4. For every pair of nonzero rows $i_1$ and $i_2$ such that $i_2>i_1$, the columns $j_1$ and $j_2$ containing those rows’ leading entries must satisfy $j_2>j_1$





> ​	:bulb: **Big picture**: Difference between **Reduced Row Echelon Form** and **Row Echelon Form**: The main difference is that it is easy to read the null space off the RREF, but it takes more work for the REF.

Applying a row operation to $A$ amounts to left-multiplying $A$ by an elementary matrix $E$. This preserves the null space, as $Av = 0 \iff EA v = 0$ (elementary matrices are invertible). Hence both $A$ and its RREF (and REF) have the same null space, and it is a simple matter to read off the null space from the RREF.





> ​	:bulb: reduced form is also called reduced echelon form. Therefore, it's better to memorize **reduced form** and **echelon form** at the same time.

$$
\text{Example of Row Echelon Form: }\\
\begin{bmatrix}
1&0&-3&0\\
0&2&2&0\\
0&0&0&1\\
0&0&0&0
\end{bmatrix}
\\\\
\text{Example of Reduced Row Echelon Form: }\\
\begin{bmatrix}
1&0&-3&0\\
0&1&2&0\\
0&0&0&1\\
0&0&0&0
\end{bmatrix}
$$



| Row Echelon Form                                             | Reduced Row Echelon Form                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| The first non-zero number from the left (the “leading coefficient”) is always to the right of the first non-zero number in the row above. | The first non-zero number in the first row (**the leading entry**) is the number 1. |
| Rows consisting of all zeros are at the bottom of the matrix. | The second row also starts with the number 1, which is further to the right than the leading entry in the first row. For every subsequent row, the number 1 must be further to the right. |
|                                                              | The leading entry in each row must be the only non-zero number in its column. |
|                                                              | Any non-zero rows are placed at the bottom of the matrix.    |





> ​	:computer: **Algorithm 3.6.** This algorithm transforms an $n\times(n+1)$ augmented matrix $M$ representing a linear system into its reduced form. At each step, $M$ refers to the *current state* of the matrix, *not the original state*.

- A. Set the row $i$ equal to 1.
- B. Set the column $j$ equal to 1. We will loop through columns 1 to $n$.
- C. Find the row $k$ with $k≥i$ for which $M_{kj}$ has the largest absolute value. If no such row exists for which $M_{kj} ≠0$ , then skip to step H.
- D. If $k≠i$ , then exchange rows $k$ and $i$ using elementary row operation(a) under Definition 3.3.
- E. Multiply row $i$ by $1/M{ij}$ . This sets the $(i,j)$ entry of $M$ to 1 using elementary row operation (b).
- F. For each row $r$, where $1  ≤r≤n$ and $r≠i$ , add $−M_{rj}$ times row $i$ to row $r$. This step clears each entry above and below row $i$ in column $j$ to 0 using elementary row operation (c).
- G. Increment $i$.
- H. If $j<n$ , increment $j$ and loop to step C.

//TODO C++ program of algorithm 3.6







## 3.3 Matrix Inverse


> ​	:pushpin: **Gauss-Jordan Elimination** : It is used to **transform a matrix into its reduced form.** But it can also be used to calculate the inverse of a matrix.

For an $n\times n\text{  matrix } M,$ 

- A. construct an $n\times 2n\text{  matrix } \tilde{M},$
- B. concatenating the identity matrix to the right of $\tilde{M}$,   (as shown below).

$$
\tilde{M}=
\left[
\begin{array}{cccc:cccc}
M_{11}&M_{12}&\cdots&M_{1n}&1&0&\cdots&0\\ 
M_{21}&M_{22}&\cdots&M_{2n}&0&1&\cdots&0\\
\vdots&\vdots&\ddots&\vdots&\vdots&\vdots&\ddots&\vdots\\
M_{n1}&M_{n2}&\cdots&M_{nn}&0&0&\cdots&1\\ 
\end{array}
\right]
\tag{3.34}
\label{augmented matrix}
$$



- C. Performing elementary row operations on the entire matrix $\tilde{M}$ until the left side $n×n$ matrix becomes the identity matrix $I_n$
- D. Then the right hand side is the **inverse**$M^{-1}$

$$
\begin{align}
&M\quad\times&I\\
&\Downarrow&\Downarrow\\
&I\quad\times&M^{-1}\\
\end{align}
$$





> ​	:computer: **Algorithm 3.12.** Gauss-Jordan Elimination. This algorithm calculates the **inverse** of an $n\times n$ matrix $M$.

- A. Construct the augmented matrix $\tilde{M}$ given in $\eqref{augmented matrix}$. Throughout this algorithm, $\tilde{M}$ refers to the current state of the augmented matrix, not the original state.
- B. Set the column $j$ equal to 1. We will loop through columns 1 to $n$.
- C. Find the row $i$ with $i ≥ j$ such that $\tilde{M}_{ij}$ has the largest absolute value. If no such row exists for which  $\tilde{M}_{ij}≠ 0$, then $M$ is **not invertible**.
- D. If $i ≠ j$, then exchange rows $i$ and $j$ **exchange 2 rows**. This is the **pivot operation** necessary to remove zeros from the main diagonal and to provide numerical stability.
- E. Multiply row $j$ by $1/\tilde{M}_{ij}$ . This sets the $(j,j)$ entry of $\tilde{M}$ to 1 **multiply a row by a nonzero scalar**.
- F. For each row $r$ where $1≤r≤n$ and $r ≠ j$, add $−\tilde{M}_{ij}$ times row $j$ to row $r$. This step clears each entry above and below row $j$ in column $j$ to 0, **add a multiple of one row to another row**.
- G. If $j<n$ , increment $j$ and loop to step C.



> ​	:pencil:**Theorem 3.14.** Let $M'$ be the $n\times n$ matrix resulting from the performance of an elementary row operation on the $n\times n$ matrix $M$. Then $M'=EM$, where $E$ is the $n\times n$ matrix resulting from the same **elementary row operation** performed on the identity matrix.



## 3.4 Determinant

> ​	:pencil: **Theorem 3.17.** Performing elementary row operations on a matrix has the following effects on the determinant of that matrix. 

- (a) Exchanging two rows **negates** the determinant. 

- (b) Multiplying a row by $a$ scalar a **multiplies** the determinant by $a$. 

- (c) Adding a multiple of one row to another row has no effect on the determinant.



> ​	:leftwards_arrow_with_hook:  **Corollary 3.18.** The determinant of a matrix having two identical rows is zero.

This is easy to understand geometrically. The rows in transpose are the column. The identical columns means they are linear dependent. How much the linear transformation is would be zero.



> ​	:pencil:**Theorem 3.20.** For any two $n\times n$ matrices $F$ and $G$, $\text{det}FG=\text{det}F\text{det}G$.



## 3.5 Eigenvalues and Eigenvectors

> ​	⭐**Big picture**: Eigenvector multiplied by the matrix, it was changed only in **magnitude** and not in **direction**.

For an $n\times n$ matrix $M$, there exist *nonzero* $n$-dimensional vectors $V_1,V_2,..,V_n$ such that
$$
MV_i=\lambda_iV_i
$$
Then, we have

$\lambda_i$ , **eigenvalues** of matrix $M$

$V_i$, **eigenvectors** of matrix $M$

**characteristic polynomial** , is the **degree** $n$ polynomial in $\lambda_{i}$. The roots of this polynomial yield the eigen-values of the matrix $M$



> ​	Formula of eigenvalues

$$
\begin{align}
MV_i&=\lambda_iV_i\\
MV_i-\lambda_iIV_i&=0\\
(M-\lambda_iI)V_i&=0
\end{align}
$$

(Tips) Because $\lambda$ is a scalar, the objective is to do arithmetic operation with $M$, therefore, $\lambda$ has to multiply $I$ becomes a matrix.



Because $V_i$ is nonzero. Therefore, $M-\lambda_iI$ must be **singular**. Therefore, the equation is:
$$
\text{det}(M-\lambda_iI)=0
$$
e.g. Calculate the determinant of following matrix
$$
\begin{align}
M&=
\begin{bmatrix}
1&1\\
3&-1
\end{bmatrix}
\\
M-\lambda I &= 
\begin{bmatrix}
1&1\\
3&-1
\end{bmatrix}
-\lambda
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\\
&=
\begin{bmatrix}
1-\lambda&1\\
3&-1-\lambda
\end{bmatrix}\\
\text{det}(M-\lambda I)&=(1-\lambda)(-1-\lambda)-3=0\\
&=\lambda^2-4=0\\
\end{align}
$$
Therefore, we have eigenvalues as followed.
$$
\lambda_1=-2,\lambda_2=2
$$
Take the values above into the equation
$$
\begin{align}
M-\lambda I
&=
\begin{bmatrix}
1-\lambda&1\\
3&-1-\lambda
\end{bmatrix}
\\
&=
\begin{bmatrix}
1-(±2)&1\\
3&-1-(±2)
\end{bmatrix}
\\
\text{and}\\
(M-\lambda I)V
&=0
\\
\text{so we have...}\\
\begin{bmatrix}
-1&1\\
3&-3
\end{bmatrix}V_1 &=
\begin{bmatrix}
0\\
0
\end{bmatrix}
\\
\begin{bmatrix}
3&1\\
3&1
\end{bmatrix}V_2 &=
\begin{bmatrix}
0\\
0
\end{bmatrix}
\end{align}
$$
As you might notice, the result of $V_1$ and $V_2$ are infinite. If desired, we should choose the eigenvector has **unit length**.
$$
\begin{align}
V_1&=a\begin{bmatrix}1\\1\end{bmatrix}
\\
V_2&=b\begin{bmatrix}1\\-3\end{bmatrix}
\end{align}
$$




> ​	:pencil: **Theorem 3.25.** The eigenvalues of a symmetric matrix $M$ having real entries are real numbers.



> ​	:pencil: **Theorem 3.26.** Any two eigenvectors associated with distinct eigenvalues of a symmetric matrix $M$ are orthogonal.





## 3.6 Diagonalization

> ​	Definition: if we can find a matrix $A$ such that $A^{-1}MA$ is a diagonal matrix, then we say that $A$ ***diagonalizes*** $M$.



> ​	:bulb:  **Big Picture** : any $n\times n$ matrix for which we can find $n$  *linearly independent eigenvectors* can be **diagonalized**.



> ​	:pencil: **Theorem 3.27.** Let $M$ be an $n\times n$ matrix having eigenvalues $λ_1,λ_2,...,λ_n$ , and suppose that there exist corresponding eigenvectors $V_1,V_2,...,V_n$ that form a linearly independent set. Then the matrix $A$ given by

$$
A=
\begin{bmatrix}
V_1&V2&\cdots&V_n
\end{bmatrix}
$$

> (i.e., the columns of the matrix $A$ are the eigenvectors $V_1,V_2,...,V_n$ diagonalizes $M$, and the main diagonal entries of the product $A^{−1}MA$ are the eigenvalues of $M$:

$$
A^{−1}MA=
\begin{bmatrix}
\lambda_1&0&\cdots&0\\
0&\lambda_2&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&\lambda_n\\
\end{bmatrix}
$$

> Conversely, if there exists an invertible matrix $A$ such that $A^{-1}MA$ is a diagonal matrix, then the columns of $A$ must be eigenvectors of $M$, and the main diagonal entries of $A^{-1}MA$ are the corresponding eigenvalues of $M$.



e.g.

