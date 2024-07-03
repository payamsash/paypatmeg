import os
import time

from tqdm.notebook import tqdm
import mne
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw

##### loading the subjects data ##### 

## create a dictionary of subjects with their files
directory = '/Users/payamsadeghishabestari/KI_MEG/KI_RS/tinmeg3' 
folders_dict = {}
for folder in sorted(os.listdir(directory)): ## iterate over folders in that directory
    f = os.path.join(directory, folder)
    if os.path.isdir(f): ## select only folders
        folders_dict[folder] = f

raw_rs_dict = {}
raw_er_dict = {}
subjects = list(folders_dict.keys())
folders = list(folders_dict.values())
for subject_id, folder in zip(subjects, folders):
    raw_er_dict[subject_id] = []
    files = sorted(os.listdir(folder))
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    for file in files:
        if 'empty_room_before' in file:
            raw_er_dict[subject_id].append(os.path.join(folder, files[1]))
        if 'empty_room_after' in file:
            raw_er_dict[subject_id].append(os.path.join(folder, files[0]))
    f_rs = os.path.join(folder, files[-1])
    raw_rs_dict[subject_id] = f_rs


## setting up parameters
verbose = False
sfreq = 250
(l_freq, h_freq) = (0.1, 80)
(l_freq_delta, h_freq_delta) = (0.5, 4)
(l_freq_theta, h_freq_theta) = (4, 8)
(l_freq_alpha, h_freq_alpha) = (8, 13)
(l_freq_beta, h_freq_beta) = (13, 30)
(l_freq_gamma, h_freq_gamma) = (30, 45)
subjects_dir = '/Applications/freesurfer/7.4.1/subjects'
fname_fsaverage_src = '/Users/payamsadeghishabestari/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-ico-5-src.fif'
src_to = mne.read_source_spaces(fname_fsaverage_src, verbose=False)
method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2
atlases = ['aparc', 'aparc.a2009s']
modes = ['mean', 'mean_flip']


for subject in tqdm(subjects[16:]): 
    
    ## reading the resting state and empty room recordings and computing noise covariance
    start_time = time.time()
    print(f'reading the MEG file of subejct {subject} ...')
    fname = raw_rs_dict[subject]
    raw = mne.io.read_raw_fif(fname=fname, preload=True, allow_maxshield=True, verbose=verbose)
    raw_er_fnames = raw_er_dict[subject]
    print(f'reading and computing noise covariance of the subject {subject} ...')
    if len(raw_er_fnames) == 0:
        noise_cov = mne.make_ad_hoc_cov(info=raw.info, std=None, verbose=verbose) # 5 fT/cm, 20 fT
    if len(raw_er_fnames) == 1:
        raw_er = mne.io.read_raw_fif(fname=raw_er_fnames[0], preload=True,
                                    allow_maxshield=True, verbose=verbose)
        noise_cov = mne.compute_raw_covariance(raw=raw_er, method='empirical', verbose=verbose)
    if len(raw_er_fnames) == 2:   
        raw_er_1 = mne.io.read_raw_fif(fname=raw_er_fnames[0], preload=True,
                                    allow_maxshield=True, verbose=verbose)
        raw_er_2 = mne.io.read_raw_fif(fname=raw_er_fnames[1], preload=True,
                                    allow_maxshield=True, verbose=verbose)
        raw_er = mne.concatenate_raws(raws=[raw_er_1, raw_er_2], verbose=verbose)
        noise_cov = mne.compute_raw_covariance(raw=raw_er, method='empirical', verbose=verbose)    
    
    # ###### Sensor Space ######

    ## resampling and filtering the data
    print('resampling and filtering the data, be patient, will last a while ...')
    raw.resample(sfreq=sfreq, verbose=verbose)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose) 

    ## creating ECG and EOG evoked responses
    ecg_evoked_meg,  ecg_evoked_grad = create_ecg_epochs(raw,
                                    verbose=verbose).average().apply_baseline(baseline=(None, -0.2),
                                    verbose=verbose).plot_joint(picks=['meg', 'grad'], show=False)
    eog_evoked_meg,  eog_evoked_grad = create_eog_epochs(raw,
                                    verbose=verbose).average().apply_baseline(baseline=(None, -0.2),
                                    verbose=verbose).plot_joint(picks=['meg', 'grad'], show=False)

    ## computing ICA and remove ECG, EOG and muscle artifacts (if any) and interpolating (if any)
    print('computing ICA (this might take a while) ...')
    ica = mne.preprocessing.ICA(n_components=0.95, max_iter=800, method='infomax',
                                random_state=42, fit_params=dict(extended=True)) 
    ica.fit(raw, verbose=verbose) 
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method="ctps", measure='zscore', verbose=verbose)
    if len(ecg_indices) > 0:
        ecg_component = ica.plot_properties(raw, picks=ecg_indices, verbose=verbose, show=False)
    emg_indices, emg_scores = ica.find_bads_muscle(raw, verbose=verbose)
    if len(emg_indices) > 0:
        emg_component = ica.plot_properties(raw, picks=emg_indices, verbose=verbose, show=False)
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG002', verbose=verbose) 
    if len(eog_indices) > 0:
        eog_component = ica.plot_properties(raw, picks=eog_indices, verbose=verbose, show=False)
    saccade_indices, saccade_scores = ica.find_bads_eog(raw, ch_name='EOG001', verbose=verbose) 
    if len(saccade_indices) > 0:
        saccade_component = ica.plot_properties(raw, picks=saccade_indices, verbose=verbose, show=False)

    exclude_idxs = ecg_indices + emg_indices + eog_indices + saccade_indices
    ica.apply(raw, exclude=exclude_idxs, verbose=verbose)
    raw.interpolate_bads(verbose=verbose)

    ## bandpassing the recording into alpha and gamma bands and saving them
    print('Bandpassing data ...')
    raw_delta = raw.copy().filter(l_freq=l_freq_delta, h_freq=h_freq_delta, verbose=verbose)
    raw_theta = raw.copy().filter(l_freq=l_freq_theta, h_freq=h_freq_theta, verbose=verbose)
    raw_beta = raw.copy().filter(l_freq=l_freq_beta, h_freq=h_freq_beta, verbose=verbose)

    fname_delta = f'/Users/payamsadeghishabestari/meg_gsp/raws/tinmeg3/delta/{subject}_raw_tsss.fif'
    fname_theta = f'/Users/payamsadeghishabestari/meg_gsp/raws/tinmeg3/theta/{subject}_raw_tsss.fif'
    fname_beta = f'/Users/payamsadeghishabestari/meg_gsp/raws/tinmeg3/beta/{subject}_raw_tsss.fif'
    raw_delta.save(fname=fname_delta, overwrite=True, verbose=verbose)
    raw_theta.save(fname=fname_theta, overwrite=True, verbose=verbose)
    raw_beta.save(fname=fname_beta, overwrite=True, verbose=verbose)

    ###### Source Space ######
    spacing = "oct6"
    if subject == "1004": spacing = "oct4"
    if subject == "1045": spacing = "ico5"
    
    ## setting up the surface source space
    print(f'Setting up bilateral hemisphere surface-based source space with subsampling for subject {subject} ...')
    src = mne.setup_source_space(subject=f'{subject}', spacing=spacing,
                                subjects_dir=subjects_dir, n_jobs=-1, verbose=True)

    ## setting up the boundary-element model (BEM) 
    print(f'Creating a BEM model for the subject ...')
    bem_model = mne.make_bem_model(subject=f'{subject}', ico=4, subjects_dir=subjects_dir, verbose=True)  
    bem = mne.make_bem_solution(bem_model, verbose=verbose)
    
    ## aligning coordinate frame (coregistration MEG-MRI)
    print(f'Coregistering MRI with a subjects head shape ...')
    coreg = Coregistration(raw_delta.info, f'{subject}', subjects_dir, fiducials='auto')
    coreg.fit_fiducials(verbose=verbose)
    coreg.fit_icp(n_iterations=40, nasion_weight=2.0, verbose=verbose) 
    coreg.omit_head_shape_points(distance=5.0 / 1000)
    coreg.fit_icp(n_iterations=40, nasion_weight=10, verbose=verbose) 
    fname_trans = f'/Users/payamsadeghishabestari/meg_gsp/trans/{subject}-trans.fif'
    mne.write_trans(fname_trans, coreg.trans, overwrite=True, verbose=verbose)
    
    ## computing the forward solution
    print(f'Computing the forward solution ...')
    fwd = mne.make_forward_solution(raw_delta.info, trans=coreg.trans, src=src, bem=bem, meg=True,
                                    eeg=False, mindist=5.0, n_jobs=None, verbose=verbose)
    
    # ## computing the minimum-norm inverse solution
    print(f'Computing the minimum-norm inverse solution ...')
    inverse_operator = make_inverse_operator(raw_delta.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=verbose)

    ## compute source estimate object
    print(f'Computing the source estimate object ...')
    for raw_bp, title_raw in zip([raw_delta, raw_theta, raw_beta], ['delta', 'theta', 'beta']):

        stc = apply_inverse_raw(raw_bp, inverse_operator, lambda2, method=method, verbose=verbose)

        # ## morphing to template brain
        # morph = mne.compute_source_morph(stc, subject_from=f'0{subject}', subject_to="fsaverage",
        #                                 subjects_dir=subjects_dir, src_to=src_to)
        # stc_morph = morph.apply(stc)
    
        ## from source estimate object to brain parcels
        for atlas, title_atlas in zip(atlases, ['aparc', 'aparc.a2009s']):
            if atlas == 'aparc':
                labels = mne.read_labels_from_annot(subject=f'{subject}', parc=atlas,
                                                    subjects_dir=subjects_dir, verbose=verbose)
            if atlas == 'aparc.a2009s':
                labels = mne.read_labels_from_annot(subject=f'{subject}', parc=atlas,
                                                    subjects_dir=subjects_dir, verbose=verbose)[:-2]

            for mode in modes:
                tcs = stc.extract_label_time_course(labels, src, mode=mode, allow_empty=True, verbose=verbose)
                file_name = f'/Users/payamsadeghishabestari/meg_gsp/stc_labels/tinmeg3/subject_{subject}_{title_raw}_{title_atlas}_{mode}.npy'
                np.save(file=file_name, arr=tcs, allow_pickle=True)

    ## creating a report for each subject
    report = mne.Report(title=f'report_subject_{subject}', verbose=verbose)

    report.add_raw(raw=raw, title='recording after preprocessing', butterfly=False, psd=False) 
    report.add_figure(fig=ecg_evoked_meg, title='ECG evoked MEG', image_format='PNG')
    report.add_figure(fig=ecg_evoked_grad, title='ECG evoked Gradiometer', image_format='PNG')
    report.add_figure(fig=eog_evoked_meg, title='EOG evoked MEG', image_format='PNG')
    report.add_figure(fig=eog_evoked_grad, title='EOG evoked Gradiometer', image_format='PNG')

    if len(ecg_indices) > 0:
        report.add_figure(fig=ecg_component, title='ECG component', image_format='PNG')
    if len(emg_indices) > 0:
        report.add_figure(fig=emg_component, title='EMG component', image_format='PNG')
    if len(eog_indices) > 0:
        report.add_figure(fig=eog_component, title='EOG component (blink)', image_format='PNG')  
    if len(saccade_indices) > 0:
        report.add_figure(fig=saccade_component, title='EOG component (saccade)', image_format='PNG') 

    report.add_bem(subject=f'{subject}', subjects_dir=subjects_dir, title="MRI & BEM", decim=10, width=512)
    report.add_trans(trans=fname_trans, info=raw.info, subject=f'{subject}',
                    subjects_dir=subjects_dir, alpha=1.0, title="Co-registration")
    
    fname_report = f'/Users/payamsadeghishabestari/meg_gsp/reports/report_subject_{subject}.html'
    report.save(fname=fname_report, open_browser=False, overwrite=True, verbose=verbose)
    print(f'elapsed time for subject {subject} was {time.time() - start_time}')