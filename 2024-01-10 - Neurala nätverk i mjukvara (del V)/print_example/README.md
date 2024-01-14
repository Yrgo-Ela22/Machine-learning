# Exempel på utskrift av tal lagrade i en vektor.

Detta exempel demonstrerar en utskriftsfunktion för att skriva ut numeriskt innehåll  
lagrat i en vektor på samma sätt som utskrift av innehållet i listor sker i Python.  
Som exempel, följande Python-kod:  

l1 = [1, 2, 3, 4, 5]  
print(l1)  

medför följande utskrift:  

[1, 2, 3, 4, 5]  

Via funktionen print definierad i filen main.cpp kan motsvarande C++kod används för  
att generera samma utskrift:  

int main()  
{  
    const std::vector<double> v1{1, 2, 3, 4, 5};  
    print(v1);  
    return 0;  
}  