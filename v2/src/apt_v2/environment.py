"""Read-only SQL environment with verified milestones and recoverable errors."""
import sqlite3
from .tasks import connect,truth,POLICY,SCHEMA

METRICS=('current_net_cents','prior_net_cents','change_cents')
TOOLS={
 'read_policy':{},'inspect_schema':{},'query_sql':{'sql':'read-only SELECT, use exact metric aliases'},
 'calculate_change':{'current_evidence':'query ID','prior_evidence':'query ID'},
 'check_result':{'values':'three integer metrics','evidence':'metric -> evidence ID','policy_id':'revenue-v1'},
 'submit':{'values':'three integer metrics','evidence':'metric -> evidence ID','policy_id':'revenue-v1'}}

class ToolError(ValueError):pass

def execute_readonly(db,sql,allowed_tables=('orders','refunds')):
    if not isinstance(sql,str) or not sql.strip() or len(sql)>5000:
        raise ToolError('provide a SELECT of at most 5000 characters')
    reads=set()
    functions={'sum','coalesce','count','min','max','abs','round','nullif','total'}
    def authorizer(op,a,b,db_name,trigger):
        if op==sqlite3.SQLITE_SELECT:return sqlite3.SQLITE_OK
        if op==sqlite3.SQLITE_READ:
            if a not in allowed_tables:return sqlite3.SQLITE_DENY
            reads.add(a);return sqlite3.SQLITE_OK
        if op==sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if (b or a or '').lower() in functions else sqlite3.SQLITE_DENY
        if op==getattr(sqlite3,'SQLITE_RECURSIVE',-1):return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    calls=0
    def progress():
        nonlocal calls
        calls+=1
        return int(calls>100)
    db.set_authorizer(authorizer);db.set_progress_handler(progress,1000)
    try:
        cursor=db.execute(sql)
        columns=[col[0] for col in cursor.description or []]
        if len(columns)!=len(set(columns)):raise ToolError('use distinct column aliases')
        rows=cursor.fetchmany(21)
        if len(rows)>20:raise ToolError('return at most 20 rows')
        if not reads:raise ToolError('query must read source tables')
        return [dict(zip(columns,row)) for row in rows],reads
    except sqlite3.Error as exc:raise ToolError(str(exc)) from exc
    finally:
        db.set_authorizer(None);db.set_progress_handler(None,0)

class Environment:
    def __init__(self,task,reward_mode='potential',tool_cost=.01):
        if reward_mode not in ('outcome','potential'):raise ValueError('unknown reward mode')
        if tool_cost<0:raise ValueError('tool cost must be nonnegative')
        self.task=task;self.db=connect(task);self.gold=truth(task)
        self.reward_mode=reward_mode;self.tool_cost=tool_cost
        self.verified=set();self.evidence={};self.records=[]
        self.policy_read=self.schema_read=self.done=self.success=False
        self.query_attempts=0;self.tool_errors=0;self.progress_after_error=False

    def potential(self):return .2*len(self.verified)

    def _certify_sql(self,sql,rows,reads):
        if len(rows)!=1:return {}
        valid={}
        for metric in METRICS:
            value=rows[0].get(metric)
            if type(value) is not int or value!=self.gold[metric]:continue
            if not {'orders','refunds'}.issubset(reads):continue
            matched=True
            for variant in (1,2):
                probe=connect(self.task,variant)
                try:
                    test,_=execute_readonly(probe,sql)
                    matched=matched and len(test)==1 and test[0].get(metric)==truth(self.task,variant)[metric]
                finally:probe.close()
            if matched:valid[metric]=value
        return valid

    def _check(self,args):
        if not isinstance(args,dict):raise ToolError('arguments must be an object')
        values=args.get('values',{});evidence=args.get('evidence',{})
        checks={'policy':self.policy_read and args.get('policy_id')=='revenue-v1','schema':self.schema_read}
        for metric in METRICS:
            value=values.get(metric) if isinstance(values,dict) else None
            eid=evidence.get(metric) if isinstance(evidence,dict) else None
            checks[metric]=(type(value) is int and value==self.gold[metric]
                and self.evidence.get(eid,{}).get(metric)==value)
        return {'passed':all(checks.values()),'checks':checks}

    def _call(self,tool,args):
        if not isinstance(args,dict):raise ToolError('arguments must be an object')
        if tool=='read_policy':
            self.policy_read=True;return {'policy_id':'revenue-v1','policy':POLICY}
        if tool=='inspect_schema':
            self.schema_read=True;return {'schema':SCHEMA}
        if tool=='query_sql':
            self.query_attempts+=1
            if self.task.fault_mode=='transient_query' and self.query_attempts==1:
                raise ToolError('temporary query service error; retry the query')
            rows,reads=execute_readonly(self.db,args.get('sql'))
            valid=self._certify_sql(args['sql'],rows,reads)
            eid='q'+str(self.query_attempts)
            self.evidence[eid]=valid
            self.verified.update(valid)
            return {'rows':rows,'evidence_id':eid,'verified_metrics':list(valid),
                'verification':'source and perturbed-data checks'}
        if tool=='calculate_change':
            cur=self.evidence.get(args.get('current_evidence'),{}).get('current_net_cents')
            prior=self.evidence.get(args.get('prior_evidence'),{}).get('prior_net_cents')
            if cur is None or prior is None:raise ToolError('provide verified monthly evidence IDs')
            value=cur-prior;eid='c'+str(len(self.evidence)+1)
            self.evidence[eid]={'change_cents':value};self.verified.add('change_cents')
            return {'change_cents':value,'evidence_id':eid,'verified_metrics':['change_cents']}
        if tool in ('check_result','submit'):
            check=self._check(args)
            if tool=='submit' and check['passed']:
                self.done=self.success=True
                return {**check,'artifact':args,'submitted':True}
            return {**check,'submitted':False}
        raise ToolError('unknown tool; choose '+', '.join(TOOLS))

    def step(self,action):
        if self.done:raise ToolError('episode already ended')
        before=self.potential();old_verified=set(self.verified);error=None
        try:
            if not isinstance(action,dict) or set(action)!={'tool','arguments'}:
                raise ToolError('action must contain exactly tool and arguments')
            observation=self._call(action['tool'],action['arguments'])
        except (ToolError,TypeError,KeyError) as exc:
            error=str(exc);self.tool_errors+=1
            observation={'error':error,'recoverable':True}
        if self.tool_errors and self.verified-old_verified:self.progress_after_error=True
        if len(self.records)+1>=self.task.max_turns:self.done=True
        after=0. if self.done else self.potential()
        # Terminal potential resets to zero, so shaping telescopes for both success and failure.
        process=after-before
        outcome=float(self.success)-self.tool_cost
        row={'turn':len(self.records),'action':action,'observation':observation,
            'potential_before':before,'potential_after':after,'process_reward':process,
            'outcome_reward':outcome,'reward':outcome+(process if self.reward_mode=='potential' else 0.),
            'new_verified_metrics':sorted(self.verified-old_verified),'done':self.done,'success':self.success,
            'tool_error':error,'termination':'success' if self.success else 'budget' if self.done else None}
        self.records.append(row)
        return row

    def close(self):self.db.close()
