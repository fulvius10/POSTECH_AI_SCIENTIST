# Fonte: IAST-ProcessamentodeLinguagemNatural-Fase04_aula02-v1.pptx
# Slide 13: REGEX EM PYTHON
# Bloco 01

import re
# Padrões básicos úteis em pré-processamento de texto
texto = "<p>Adorei!! https://link.com vc tem que ver isso :)</p>"
# Remover tags HTML
limpo = re.sub(r'<[^>]+>', ' ', texto)
# 'Adorei!! https://link.com vc tem que ver isso :)'
# Remover URLs
limpo = re.sub(r'http\S+|www\.\S+', ' ', limpo)
# 'Adorei!!  vc tem que ver isso :)'
# Remover e-mails
limpo = re.sub(r'\S+@\S+', ' ', limpo)
# Manter apenas letras (incluindo acentuadas) e espaços
limpo = re.sub(r'[^a-záàâãäéèêëíìîïóòôõöúùûüç\s]', ' ', limpo)
# 'Adorei   vc tem que ver isso  '
# Colapsar espaços múltiplos
limpo = re.sub(r'\s+', ' ', limpo).strip()
# 'Adorei vc tem que ver isso'
