## Instala webdriver

Minha versão: Versão 137.0.7151.41 (Versão oficial) 64 bits
Baixar versão do webdriver: 

- Chrome: https://storage.googleapis.com/chrome-for-testing-public/137.0.7151.41/win64/chrome-win64.zip
- Chromedriver: https://storage.googleapis.com/chrome-for-testing-public/137.0.7151.41/win64/chromedriver-win64.zip


O ChromeDriver é um executável separado que o Selenium WebDriver usa para controlar o Chrome. Ele é mantido pela equipe do Chromium, com a ajuda dos colaboradores do WebDriver. Se você não conhece o Selenium WebDriver, acesse o site do Selenium.

Siga estas etapas para configurar seus testes para execução com o ChromeDriver:

- Verifique se o Chromium/Google Chrome está instalado em um local reconhecido
- O ChromeDriver espera que o Chrome esteja instalado no local padrão da plataforma. Você também pode forçar o ChromeDriver a usar um local personalizado configurando um recurso especial.
- Faça o download do binário do ChromeDriver para sua plataforma na seção downloads deste site
- Ajudar o WebDriver a encontrar o executável do ChromeDriver baixado

Qualquer uma destas etapas deve ajudar:

1. inclua o local do ChromeDriver na variável de ambiente PATH
2. (somente Python) inclua o caminho para o ChromeDriver ao instanciar o webdriver.Chrome (veja o exemplo abaixo).

## Exemplo de teste

import time
from selenium import webdriver

driver = webdriver.Chrome('/path/to/chromedriver')  # Optional argument, if not specified will search path.
driver.get('http://www.google.com/');
time.sleep(5) # Let the user actually see something!
search_box = driver.find_element_by_name('q')
search_box.send_keys('ChromeDriver')
search_box.submit()
time.sleep(5) # Let the user actually see something!
driver.quit()