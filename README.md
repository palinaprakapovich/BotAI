# BotAI
Aplikacja do wykrywania ptaków.


Nadal nie wiesz jak odróżnić gołębia od wróbla?
Mój bot może Ci w tym pomóc! 
Moja aplikacja potrafi odróżnic gołębia od wróblia, podać z jaką dokładnością on to zrobił i podać ciekawy fakt, dotyczący ptaka ze zdjęcia. 
Na razie ten bot może wykrywać tylko dwa rodzaje ptaków, więc przysyłaj mu zdjęcia tych golębia albo wróbla. 


Instrukcja do korzystania z mojej aplikacji jest bardzo latwa:

1.Pobierz wszystkie pliki
![image](https://github.com/user-attachments/assets/cb06e1ba-d9e3-454f-9869-995abcc653d4)

2.Otwórz te pliki w folderze
![image](https://github.com/user-attachments/assets/a0281163-33fc-4252-890f-9eb4d4f909da)

3.Otwórz ten folder w VSC, PyCharm albo innej aplikacji do programowania
![image](https://github.com/user-attachments/assets/877f5430-626a-4b22-984b-74624c603fbb)

4.Utwórz środowisko wirtualne w folderze. Użyj polecenia pipenv shell, aby utworzyć środowisko wirtualne w folderze.
![image](https://github.com/user-attachments/assets/f070c0df-f99c-4de5-b9fa-f7609a7cf630)

5.Po utworzeniu środowiska wirtualnego, musisz je wybrać. Aby to zrobić, naciśnij Shift + Ctrl + P -> Select Interpretator i znajdź opcję, która ma taką samą nazwę jak twój projekt
![image](https://github.com/user-attachments/assets/959ac5f2-fdab-4235-b1b8-9ee0e7647127)
![image](https://github.com/user-attachments/assets/84a5ea15-ca0c-414f-aa8a-c94a554bd9dc)

Aby upewnić się, że wszystko działa zgodnie z zamierzeniem, sprawdź dwie rzeczy:

1)Projekt powinien teraz zawierać plik Pipfile

2)Nazwa twojego projektu powinna pojawić się w terminalu (znajdziesz ją w nawiasach)
![image](https://github.com/user-attachments/assets/64a37046-edb8-4be9-bb68-9eb870592ad2)


5.Zaimportuj biblioteki: discord.py, numpy, requests, Pillow, keras i tenserflow 
![image](https://github.com/user-attachments/assets/daa9ce78-83d3-4180-a449-5169ef866cf5)
![image](https://github.com/user-attachments/assets/644da770-da70-4e8f-a27b-f0c24923615a)
![image](https://github.com/user-attachments/assets/25e8758a-fa1f-404f-bdbd-a4cdf898cb41)

6.Za pomocą komendy pip freeze sprawdż wersje bibliotek keras i tensorflow. Powinny to być wersje 2.12.0

7. Jeżeli wersja jest inna:
 -pip uninstall keras
 -pip uninstall tensorflow
 -pip install keras == 2.12.0
 -pip install tensorflow == 2.12.0
  
8.Uruchom main.py

9.Załóż darmowe konto albo wejdż już na istniejące konto discord

10.Wyślij zdjęcie albo gołębia, albo wróbla

11.Napisz $check do zdjęcia

Jak zrobisz wszystko poprawnie będzie to wyglądało w taki sposób:
![image](https://github.com/user-attachments/assets/58791aa5-4491-48ea-a23b-3efe57846275)



