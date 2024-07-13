## declare an array variable
declare -a arr=("all" "random" "fisher")

## now loop through the above array
for i in "${arr[@]}"
do
if [ $i = all ]; then
    echo "The strings are equal."
else
    echo "The strings are not equal."
fi
done
