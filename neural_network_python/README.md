# Neural network implementation in Python

Enkelt neuralt nätverk implementerat i Python.  
Nätverket består av två ingångar, en utgång samt ett dolt lager bestående av två noder. 
I det dolda lagret används tanh som aktiveringsfunktion, medan ReLU används i utgångslagret.  

Nätverket tränas till att detekteras ett två-bitars XOR-mönster, där AB utgör indata och X utgör utdata.

|   AB   |   X   |   
| :----: | :---: |  
|   00   |   0   |  
|   01   |   1   |  
|   10   |   1   |  
|   11   |   0   |  

Nätverket tränas under 1000 epoker med en lärhastighet på 10 %.  
Efter träning predikterar nätverket med 100 % precision.