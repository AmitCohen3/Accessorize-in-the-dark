import utils
from arg_parser import Parser
from load_model import load_model
from attacks import perform_attack
from predict import predict
from utils import init_plots_dir, prepare_data_paths, save_configuration


def main():
    args = Parser().get_args()
    model = load_model(args)
    init_plots_dir()
    save_configuration(args)
    protocols_path = args.protocols_folder_name
    gallery_file_path, probe_file_path = prepare_data_paths(args.dataset_path, protocols_path, args.gallery_index)

    print("Started! Plots will be in", utils.plots_dir)
    if args.predict:
        predict(args, model, gallery_file_path, probe_file_path)
    else:
        acc_before_attack, acc_after_attack, target_att_succ_rate = perform_attack(args, model, gallery_file_path, probe_file_path)
        print(f"Accuracy before attack: {acc_before_attack} || Accuracy after attack: {acc_after_attack} || Targeted attack success rate: {target_att_succ_rate}")
    print("Finished! Plots are in ", utils.plots_dir)

if __name__ == '__main__':
    main()
