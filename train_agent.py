from tensorforce.core import ArrayDict
import numpy as np
import os, shutil


def prepare_training_data(ep_all, ep_train, batch_size):    
    files_all = sorted(
        os.path.join(ep_all, f) for f in os.listdir(ep_all)
        if os.path.isfile(os.path.join(ep_all, f)) and os.path.splitext(f)[1] == '.npz'
    )
    os.system('rm -rf ' + ep_train + '/*')
    for f in files_all[-batch_size:]:
        shutil.copyfile(f, f.replace(ep_all,ep_train))


def experience_update(agent, directory, extension='.npz'):
    """
        Training approach as a combination of `experience()` and `update`
        Feed the recorded experience of the whole batch to the agent and update it        
        The function is inspired by the pretrain funciton of tensorforce
    """    
    files = sorted(
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1] == extension
    )
    indices = list(range(len(files)))

    batch = None
    for index in indices:
        trace = ArrayDict(np.load(files[index]))
        if batch is None:
            batch = trace
        else:
            batch = batch.fmap(
                function=(lambda x, y: np.concatenate([x, y], axis=0)), zip_values=(trace,)
            )

    for name, value in batch.pop('auxiliaries', dict()).items():
        assert name.endswith('/mask')
        batch['states'][name[:-5] + '_mask'] = value

    agent.experience(**batch.to_kwargs())
    agent.update()
