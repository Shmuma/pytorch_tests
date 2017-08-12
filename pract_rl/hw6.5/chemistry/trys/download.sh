#!/bin/sh
test -f eng-fra.txt && exit 0

wget http://www.manythings.org/anki/fra-eng.zip
unzip fra-eng.zip fra.txt
rm fra-eng.zip
mv fra.txt eng-fra.txt
