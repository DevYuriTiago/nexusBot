# NEXUS - Acelerador de Conteúdo com IA

NEXUS é um assistente de IA que ajuda a criar conteúdo viral para redes sociais, especialmente Instagram. Ele busca conteúdo relevante no Reddit, analisa as tendências e gera:

- Carrossel de posts otimizados
- Legendas envolventes
- Imagens geradas por IA

## Tecnologias

- Python 3.12+
- Flask
- Transformers (Phi-2 para geração de texto)
- PRAW (API do Reddit)
- Kandinsky (geração de imagens)

## Configuração

1. Clone o repositório
2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente no arquivo `.env`:
```
REDDIT_CLIENT_ID=seu_client_id
REDDIT_CLIENT_SECRET=seu_client_secret
HUGGINGFACE_TOKEN=seu_token
```

5. Execute o servidor:
```bash
python bot.py
```

## Estrutura do Projeto

```
Bot/
├── bot.py              # Servidor Flask e lógica principal
├── templates/          # Templates HTML
│   └── index.html      # Interface do usuário
├── requirements.txt    # Dependências do projeto
├── .env               # Variáveis de ambiente (não versionado)
└── README.md          # Este arquivo
```

## Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request
