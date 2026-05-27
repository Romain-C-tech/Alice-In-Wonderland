<h1 align="center">Alice In Wonderland</h1>
<div align="center">
    <img src= "https://upload.wikimedia.org/wikipedia/commons/4/42/The_White_Rabbit_%28Tenniel%29_-_The_Nursery_Alice_%281890%29_-_BL.jpg" alt="The White Rabbit jpg" width="150"/>
</div>
<p>
    We've just joined Through the Looking-Glass, a startup building a lightweight tool that creates "book cards" to help publishers and editors quickly make sense of Wonderland's endless library, without having to read the books entirely.
</p>


## Clone the repository
```bash
mkdir NLP_Engine
cd NLP_Engine

git clone git@github.com:EpitechBachelorPromo2028 B-AIA-200-NAN-2-1-aliceinwonderland-1.git
```

## Installation

Our NLP is written in Python and needs some libraries to be installed before using it.
```bash
pip install -r requirements.txt
```
## Manual

Our NLP can be used in different ways that can be seen by using this command in the terminal.
```bash
python3 bookworm.py --help
```
To use correctly our different options you **MUST** follow this order :
```bash
python3 bookworm.py --options value
```
The order is important and will cause error if its not called in this way.

## Ressources

For our data, we use the catalog of book given by The Project Gutenberg and also its API to have all the text in plain text format.