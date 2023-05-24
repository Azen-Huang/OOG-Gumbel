for ((i = 95; i <= 200; i = i + 1))
do
    echo "iter : $i Start"
    g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --self_play $((i))    
    python -u "./train_network.py"

    if ((i % 10 == 0 ))
    then
        #g++ -march=native KataGo.cc -g -ltensorflow -fopenmp -O2 -o KataGo && "./"KataGo --evaluation $i | tee output_$((i))_vs_$((i - 10)).log
        g++ -march=native Gumbel.cc -g -ltensorflow -fopenmp -O2 -o Gumbel && "./"Gumbel --evaluation $i | tee output_$((i))_vs_$((i - 10)).log
        python -u "./copy_folder.py" -1
        python -u "./copy_folder.py" $i
    fi

    if((i >= (4) && i <= (35) && i % 2 == 0))
    then
        python -u "./del_history.py"
    fi

    if((i > (35))) #6 file 4 file => 3 + 40
    then 
        python -u "./del_history.py"
    fi
done
