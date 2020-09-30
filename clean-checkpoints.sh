#!/bin/bash
echo "The next chekpoint folders will be deleted (all but 5 more recent):"
ls -dt models/bertsification-*/checkpoint*/ | awk 'NR>5'
read -p "Do you want to continue? (y/n) " RESP
if [ "$RESP" = "y" ]; then
  echo "Deleting..."
  rm -rf `ls -dt models/${PREFIX-bertsification}-*/checkpoint*/ | awk 'NR>5'`
  echo "Done"
else
  echo "Nothing was deleted"
fi
