# 2023-10-13 - Neurala nätverk i mjukvara (del IV)
Implementering av dense-lager i C++ - del IV.  

Algoritmer för feedforward, backpropagatation och optimering samt unit-test av dense-lager har blivit implementerat.  
Nästa steg, som genomförs under nästa termin, består av att skapa en modul för neurala nätverk via den implementerade dense-lager-modulen.  

Implementeringen har förfinats något, primärt via uppdelning av ett fåtal funktioner för enkel unit testing samt användning av aktiveringsfunktioner.
De körbara filerna har placerats i separata output-kataloger. 

Shell-skriptet "make.sh" implementeras för att kompilera koden och köra programmet direkt från baskatalogen "neural_network".  