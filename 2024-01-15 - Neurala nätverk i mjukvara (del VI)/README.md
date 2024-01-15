# 2023-01-15 - Neurala nätverk i mjukvara (del VI)
Slutförd implementering av neuralt nätverk i C++.

Nätverket består av två ingångar, en utgång samt ett dolt lager bestående av tre noder.   
I det dolda lagret används tanh som aktiveringsfunktion, medan ReLU används i utgångslagret.  

Nätverket tränas tränas till att prediktera att 2-bits XOR-mönster såsom visas nedan,   
där AB utgör indata och X utgör nätverkets utdata:

|   AB   |   X   |   
| :----: | :---: |  
|   00   |   0   |  
|   01   |   1   |  
|   10   |   1   |  
|   11   |   0   |  


Träning sker under 10 000 epoker med en lärhastighet på 1 %.  
Efter träning predikterar nätverket med 100 % precision.  