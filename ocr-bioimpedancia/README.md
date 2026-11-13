# OCR de bioimpedância · Body-composition OCR

Lê a imagem de um laudo de bioimpedância com Tesseract, extrai os campos (peso,
água corporal, proteína, gordura, massa muscular, TMB, IMC…) com expressões
regulares ajustadas ao ruído do OCR e grava o resultado numa planilha Excel,
criando aba e cabeçalhos quando não existem.

*Reads a body-composition report image with Tesseract, extracts the fields (weight,
body water, protein, fat, muscle mass, BMR, BMI…) with regexes tuned to OCR noise
and appends the result to an Excel sheet, creating the tab and headers when needed.*

Caminhos e credenciais vêm de variáveis de ambiente: `IMAGEM_EXAME`,
`EXCEL_DESTINO`, `PLANILHA_DESTINO`, `GOOGLE_APPLICATION_CREDENTIALS`.

`pytesseract` · `Pillow` · `re` · `openpyxl` · `gspread`
