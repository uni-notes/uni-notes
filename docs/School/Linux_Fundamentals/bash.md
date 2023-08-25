1. Read - Takes Input, '$' sign for variables.
    ```
    read name
    echo ("Welcome $name")
    ```

2. For Loops: Have two braces for Math Ops (())
    ```
    for ((i = 1; i <= 100; i+=2))
    do
        $i
    done
    ```

3. While Loop Structure - do, (*stuff*), done

4. We use " | bc" for math operations. 
    ```
    echo("5+3") | bc
    ```

5. Conditionals : if, elif, else, fi

    ```
    read input

    if [ $input == "y" ] || [ $input == "Y" ]; then 
        echo "YES"
    else 
        echo "NO"
    fi
    ```

6. grep : Will search and print matching. 

    ```
    grep -iwE "the|that|then|those" 

    # i - To search case-insensitive. 
    # w - To ignore case. 
    # E - To compare as an extended regex, to allow use of "|"

    grep -v "that" 
    # v - Will show invert of what you're searching.
    ```
7. sed : To replace. 

    ```
    sed 's/the /this /1'
    #Space to avoid words like "therefore"

    sed -e 's/thy/your/ig' -e 's/Thy/Your/ig' -e 's/tHy/YouR/ig'
    #To replace mutliple words with one
    # -e is for editing

    sed -e 's/thy/{&}/gi'
    #Make it highlight in curly braces. 

    sed -e 's/([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)/\4 \3 \2 \1/' 
    #Back-Refrencing. 
    #E - for extended regular expression 
    ```

8. awk - For comparision and parsing.
    ```
    awk '{ if ($4 == "") print "Not all scores are available for "$1}'
    Variable out of the "". 
    
    awk '{ if (($2 >= 50) && ($3 >= 50) && ($4 >=50)) {print $1" : Pass"} else {print $1" : Fail"}}'
    

    awk '{
    total = ($2 + $3 + $4)/3
    if (total >= 50 && total< 60)
        print $1,$2,$3,$4, ": C"
    else if (total >= 60 && total < 80)
        print $1,$2,$3,$4, ": B"
    else if (total >= 80)
        print $1,$2,$3,$4, ": A" 
    else 
        print $1,$2,$3,$4, ": FAIL" }'


    awk '
        BEGIN {count=0}
        
        {
            printf "%s %d %d %d", $1, $2, $3, $4
            count++;
            if (count%2 == 0)
                printf "\n"  
            else
                printf ";"
        }
    '
    ```

9. Arrays: 

    ```
    while read country
    do
        countries=("${countries[@]}" "$country")
    done
    echo "${countries[@]}"
    # First one is array, second one is element being added to array. 


    while read country
    do 
        countries+=($country)
    done
    echo "${countries[@]:3:5}"
    #Sliced

    while read country
    do 
        if [[ "$country" != *a* && "$country" != *A* ]]; then
            countries+=("$country")
        fi
    done
    echo "${countries[@]}"
    # Filterting on the basis of letter. 

    while read line
    do
        elements+=(i)
    done 
    echo ${#elements[@]}
    # counts - "#"
    ```

10. Cut

    ```
    while read lines
    do
        echo "$lines" | cut -c3
    done
    # Cut a specifc character. 

    while read lines
    do
        echo "$lines" | cut -c 2,7
    done
    # Cut multiple characters. 

    while read lines
    do
        echo "$lines" | cut -c 2-7
    done
    # Cut a range of charavters. 

    while read lines 
    do 
        echo "$lines" | cut -d $'\t' -f 1-3
    done
    # -d : only takes one character so use 'X'

    while read lines 
    do 
        echo $lines | cut -c 13-
    done
    # Print from X character to the end. 

    while read line
    do
        echo "${line}" | cut -d ' ' -f 4
    done
    # 4th Word, delimeter is ' '.

    ```

11. Head and Tail: 

    ```
    head -n 20 
    # For first 20 lines of text. 

    head -c 20 
    # For first 20 characters of text. 

    head -n 22 | tail -n +12
    # First 22 characters && Prints from 12th to last (22nd Character)
    ```

12. tr and sort : tr replaces, sort sorts (lol)

    ```
    tail -n 20
    # Last 20 lines.

    tr '()' '[]'
    # Replace. 

    tr -d "(a-z)"
    # Delete all lower-case characters. 

    tr -s ' '
    # Squeezes spaces.

    sort -f
    # Sorts in alphabetical order, -f ignore case. 

    sort -r
    # Sorts in reverse alphabetical order. 

    sort -n 
    # Sorts in numerically ascending order. 

    sort -n -r
    # Sorts in numerically descending order. 

    sort -r -n -k 2 -t $'\t'
    # -r : For reverse order 
    # -n : numerical sort 
    # -k: column ordering 
    # -t : tab separted indicator
    ```

13. Uniq: 
    ```
        uniq
        # Eliminates repetitions. 

        uniq -c 
        # Counts the repitions. 
    ```
14. Paste:
    ```
    paste -d'\t' -s
    # '-s' tells the command to write everything sequentially in one line. 

    paste -s -d ';'
    # Pastes with a delimeter specified. 
    ```