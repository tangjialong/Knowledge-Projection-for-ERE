import argparse

def view(model1_output_path, output_file_path):
    with open(output_file_path, 'w') as output_file:
        model1_output = open(model1_output_path)
        output_file.write("{}\t{}\t{}\t{}\t{}\n".format("GUID", "TEXT", "GOLDEN_TAG", "MODEL1_PRED", "STATE"))
        for instance1 in model1_output:
            instance1 = instance1.strip().split('\t')
            if float(instance1[3]) > float(instance1[5]):
                continue
            else:
                output_file.write("{}\t{}\t{}\t{}\t{}\n".format(instance1[0],instance1[1],instance1[2],instance1[4],'Wrong'))
        output_file.close()


def compare(model1_output_path, model2_output_path, output_file_path):
    with open(output_file_path, 'w') as output_file:
        model1_output = open(model1_output_path)
        model2_output = open(model2_output_path)

        output_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("GUID", "TEXT", "GOLDEN_TAG", "MODEL1_PRED", "MODEL2_PRED", "DIFFERENCE"))
        for instance1, instance2 in zip(model1_output, model2_output):
            instance1 = instance1.strip().split('\t')
            instance2 = instance2.strip().split('\t')
            guid = instance1[0]
            text = instance1[1]
            golden_tag = instance1[2]

            if float(instance1[3]) < float(instance1[5]):
                instance1_tag = 'Good'
                instance1_pred = instance1[2]
            else:
                instance1_tag = 'Bad'
                instance1_pred = instance1[4]
            if float(instance2[3]) < float(instance2[5]):
                instance2_tag = 'Good'
                instance2_pred = instance2[2]
            else:
                instance2_tag = 'Bad'
                instance2_pred = instance2[4]
            if instance1_pred != instance2_pred:
                output_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(guid,text,golden_tag,instance1_pred,instance2_pred,instance1_tag+'->'+instance2_tag))
            
        model1_output.close()
        model2_output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--MODE', help='MODE: view or compare.', default='v')
    parser.add_argument('-m1', '--MODEL1_OUTPUT_PATH', help='PATH of MODEL1 OUTPUT.', default='./models/Done/BERT-large.tsv')
    parser.add_argument('-m2', '--MODEL2_OUTPUT_PATH', help='PATH of MODEL2 OUTPUT.', default='./models/BERT-large.tsv')
    parser.add_argument('-o', '--OUTPUT_FILE_PATH', help='PATH of Output File.', default='./view.tsv')
    args = parser.parse_args()

    if args.MODE == 'v':
        view(args.MODEL1_OUTPUT_PATH, args.OUTPUT_FILE_PATH)
    elif args.MODE == 'c':
        compare(args.MODEL1_OUTPUT_PATH, args.MODEL2_OUTPUT_PATH, args.OUTPUT_FILE_PATH)
    else:
        print ('Not')