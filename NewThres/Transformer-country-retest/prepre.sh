
# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
echo "sleep start"
echo "sleep end"

t2t-trainer --registry_help

PROBLEM=translate_enzh_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu

HOME=/data/szy/FT
DATA_DIR=$HOME/t2t_d
TMP_DIR=$HOME/t2t_dg
TRAIN_DIR=$HOME/t2t_t/$PROBLEM/$MODEL-$HPARAMS

export CUDA_VISIBLE_DEVICES=3

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
#t2t-datagen \
#  --data_dir=$DATA_DIR \
#  --tmp_dir=$TMP_DIR \
#  --problem=$PROBLEM

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
#t2t-trainer \
#  --data_dir=$DATA_DIR \
#  --problem=$PROBLEM \
#  --model=$MODEL \
#  --train_steps=1000000 \
#  --hparams_set=$HPARAMS \
#  --output_dir=$TRAIN_DIR

# zh_decomma.zh
#DECODE_FILE=pad/Mu.en
DECODE_FILE=./en_mu.txt #NewThres/TestGenerator-NMT/en_mu.txt #Mu.en
#echo "Hello world" >> $DECODE_FILE
#echo "Goodbye world" >> $DECODE_FILE
#echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de

BEAM_SIZE=15
ALPHA=0.6

t2t-decoder \
      --data_dir=$DATA_DIR \
        --problem=$PROBLEM \
          --model=$MODEL \
            --hparams_set=$HPARAMS \
              --output_dir=$TRAIN_DIR \
                --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,write_beam_scores=False,return_beams=False" \
                  --decode_from_file=$DECODE_FILE \
                    --decode_to_file=./f_en_mu.zh.beamtttt #Mu.zh.beam

exit
# zh_decomma.zh
DECODE_FILE=cross/en_cross.en
#echo "Hello world" >> $DECODE_FILE
#echo "Goodbye world" >> $DECODE_FILE
#echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de

BEAM_SIZE=15
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,write_beam_scores=True,return_beams=True" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=cross/cross.zh


