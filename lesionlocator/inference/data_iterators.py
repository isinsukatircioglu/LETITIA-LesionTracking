import multiprocessing
import queue
from torch.multiprocessing import Event, Queue, Manager

from time import sleep
from typing import List
import torch

from lesionlocator.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from lesionlocator.utilities.prompt_handling.prompt_handler import get_prompt_from_inst_or_bin_seg, get_prompt_from_json



def preprocess_fromfiles_save_to_queue(input_files: List[str],
                                       prompt_files: List[str],
                                       output_files: List[str],
                                       prompt_type: str,
                                       plans_manager: PlansManager,
                                       dataset_json: dict,
                                       configuration_manager: ConfigurationManager,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False):
    try:
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(input_files)):
            if prompt_files[idx].endswith('.json'):
                data, _, data_properties = preprocessor.run_case([input_files[idx]],
                                                                None,
                                                                plans_manager,
                                                                configuration_manager,
                                                                dataset_json)
                
                prompt = get_prompt_from_json(prompt_files[idx], prompt_type, data_properties, data.shape[1:])
            else:
                data, prompt, data_properties = preprocessor.run_case([input_files[idx]],
                                                                prompt_files[idx],
                                                                plans_manager,
                                                                configuration_manager,
                                                                dataset_json)
                prompt = get_prompt_from_inst_or_bin_seg(prompt, prompt_type)
 
            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {'data': data, 'prompt': prompt, 'data_properties': data_properties, 'ofile': output_files[idx]}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        # print(Exception, e)
        abort_event.set()
        raise e


def preprocessing_iterator_fromfiles(input_files: List[str],
                                     prompt_files: List[str],
                                     output_files: List[str],
                                     prompt_type: str,
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(input_files), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
                     args=(
                         input_files[i::num_processes],
                         prompt_files[i::num_processes],
                         output_files[i::num_processes],
                         prompt_type,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # import IPython;IPython.embed()
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]
