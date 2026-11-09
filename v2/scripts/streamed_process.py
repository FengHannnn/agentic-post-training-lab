"""Stream child stdout/stderr to terminal and UTF-8 log while enforcing a deadline."""
import codecs,os,signal,subprocess,sys,threading

def _stop(proc):
    if proc.poll() is not None:return
    if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
    else:proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
        else:proc.kill()
        proc.wait()

def run_logged(command,*,cwd,env,log_path,timeout):
    if timeout<=0:raise TimeoutError('stage has no remaining time')
    errors=[]
    with log_path.open('x',encoding='utf8',newline='') as log:
        proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0,start_new_session=os.name!='nt')
        def copy_output():
            decoder=codecs.getincrementaldecoder('utf-8')('replace')
            try:
                while True:
                    chunk=proc.stdout.read(4096)
                    text=decoder.decode(chunk,final=not chunk)
                    if text:
                        log.write(text);log.flush()
                        try:sys.stdout.write(text);sys.stdout.flush()
                        except BrokenPipeError:pass
                    if not chunk:break
            except Exception as exc:errors.append(exc)
        reader=threading.Thread(target=copy_output,name='stage-output',daemon=True);reader.start()
        try:code=proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            _stop(proc);raise
        finally:
            reader.join(timeout=5)
            if not reader.is_alive():proc.stdout.close()
        if reader.is_alive():raise RuntimeError('stage output did not close after process exit')
        if errors:raise RuntimeError('stage output streaming failed') from errors[0]
        return code
