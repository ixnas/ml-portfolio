n=0
for file in ./*.png; do
   test $n -eq 0 && mv "$file" selected/
   n=$((n+1))
   n=$((n%4))
done
