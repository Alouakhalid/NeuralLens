import os,json,time,uuid,secrets,threading,traceback,io,base64,psutil
from datetime import datetime,timedelta
from functools import wraps
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from flask import Flask,request,jsonify,send_from_directory,Response
from flask_cors import CORS

BASE=os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(BASE,"epoch_03_valloss_0.0626.keras")
VOCAB_PATH=os.path.join(BASE,"caption_tokenizer-vocab.json")
MERGES_PATH=os.path.join(BASE,"caption_tokenizer-merges.txt")
DB_PATH=os.path.join(BASE,"neurallens_db.json")
SAMPLES=os.path.join(BASE,"training_samples")
os.makedirs(SAMPLES,exist_ok=True)
SZ=224;MX=40;PAD=0;ST=1;ED=2;UNK=3

model=None;vocab={};id2tok={};merges=[];_optimizer=None
tlock=threading.Lock()
tstatus={"active":False,"progress":0,"msg":"","log":[]}
gpu_info={"name":"CPU","avail":False}
MODEL_NAME="VisionMind Alpha"
MODEL_VERSION="1.0.0"
MODEL_DESC="Advanced ViT+GPT Image Captioning Engine"

class _ViTEncoder(keras.layers.Layer):
    def __init__(self,num_layers=6,D=768,num_heads=12,mlp_dim=3072,dropout=0.1,num_patches=196,**kwargs):
        super().__init__(**kwargs)
        self.patch_embed=keras.layers.Dense(D,name='patch_embed')
        # cls_token and pos_embedding were Embedding layers in original model
        self.cls_token=keras.layers.Embedding(1,D,name='cls_token')
        self.pos_embedding=keras.layers.Embedding(num_patches+1,D,name='pos_embedding')
        self.attention_layers=[keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=D//num_heads,name=f'multi_head_attention{"" if i==0 else "_"+str(i)}') for i in range(num_layers)]
        self.mlp_layers=[keras.Sequential([keras.layers.Dense(mlp_dim,activation='gelu'),keras.layers.Dense(D)]) for _ in range(num_layers)]
        self.norm1=[keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.norm2=[keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.norm_final=keras.layers.LayerNormalization(name='norm_final')
        self.drop=keras.layers.Dropout(dropout);self.num_layers=num_layers
    def call(self,x,training=False):
        B=tf.shape(x)[0]
        x=self.patch_embed(x)                                      # (B,196,D)
        cls=self.cls_token(tf.zeros((B,1),dtype=tf.int32))         # (B,1,D)
        x=tf.concat([cls,x],axis=1)                                # (B,197,D)
        pos_ids=tf.range(tf.shape(x)[1])[tf.newaxis]              # (1,197)
        x=x+self.pos_embedding(pos_ids)
        for i in range(self.num_layers):
            x=self.norm1[i](x);x=x+self.drop(self.attention_layers[i](x,x),training=training)
            x=self.norm2[i](x);x=x+self.drop(self.mlp_layers[i](x),training=training)
        return self.norm_final(x)

class _GPTDecoder(keras.layers.Layer):
    def __init__(self,num_layers=6,D=768,num_heads=12,mlp_dim=3072,dropout=0.1,vocab_size=8720,max_len=40,**kwargs):
        super().__init__(**kwargs)
        self.token_embedding=keras.layers.Embedding(vocab_size,D,name='token_embedding')
        self.pos_embedding=keras.layers.Embedding(max_len,D,name='pos_embedding')
        self.masked_attention=[keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=D//num_heads,name=f'multi_head_attention{"" if i==0 else "_"+str(i)}') for i in range(num_layers)]
        self.cross_attention=[keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=D//num_heads,name=f'multi_head_attention{"" if i==0 else "_"+str(i)}') for i in range(num_layers)]
        self.ffn=[keras.Sequential([keras.layers.Dense(mlp_dim,activation='gelu'),keras.layers.Dense(D)]) for _ in range(num_layers)]
        self.norm1=[keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.norm2=[keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.norm3=[keras.layers.LayerNormalization() for _ in range(num_layers)]
        self.norm_final=keras.layers.LayerNormalization(name='norm_final')
        self.output_layer=keras.layers.Dense(vocab_size,name='output_layer')
        self.drop=keras.layers.Dropout(dropout);self.num_layers=num_layers
    def call(self,token_ids,enc_out,training=False):
        seq_len=tf.shape(token_ids)[1]
        pos_ids=tf.range(seq_len)[tf.newaxis]
        x=self.token_embedding(token_ids)+self.pos_embedding(pos_ids)
        i_idx=tf.range(seq_len)[:,tf.newaxis];j_idx=tf.range(seq_len)[tf.newaxis,:]
        mask=tf.cast(i_idx<j_idx,tf.float32)*-1e9
        for i in range(self.num_layers):
            x=x+self.drop(self.masked_attention[i](x,x,attention_mask=mask),training=training);x=self.norm1[i](x)
            x=x+self.drop(self.cross_attention[i](x,enc_out),training=training);x=self.norm2[i](x)
            x=x+self.drop(self.ffn[i](x),training=training);x=self.norm3[i](x)
        return self.output_layer(self.norm_final(x))



class ImageCaptioningModel(keras.Model):
    def __init__(self,image_size=224,patch_size=16,num_patches=196,D=768,num_heads=12,num_layers=6,mlp_dim=3072,dropout=0.1,vocab_size=8720,max_len=40,batch_size=32,epochs=20,save_path='captioning_model.keras',max_len_input=39,learning_rate=0.0001,**kwargs):
        super().__init__(**kwargs)
        self.image_size=image_size;self.patch_size=patch_size;self.num_patches=num_patches
        self.D=D;self.num_heads=num_heads;self.num_layers=num_layers;self.mlp_dim=mlp_dim
        self.dropout=dropout;self.vocab_size=vocab_size;self.max_len=max_len
        self.batch_size=batch_size;self.epochs=epochs;self.save_path=save_path
        self.max_len_input=max_len_input;self.learning_rate=learning_rate
        self.encoder=_ViTEncoder(num_layers=num_layers,D=D,num_heads=num_heads,mlp_dim=mlp_dim,dropout=dropout,num_patches=num_patches,name='encoder')
        self.decoder=_GPTDecoder(num_layers=num_layers,D=D,num_heads=num_heads,mlp_dim=mlp_dim,dropout=dropout,vocab_size=vocab_size,max_len=max_len,name='decoder')
    def call(self,inputs,training=False):
        patches,seq_ids=inputs  # patches:(B,196,768) seq_ids:(B,seq)
        enc_out=self.encoder(patches,training=training)  # (B,197,768)
        return self.decoder(seq_ids,enc_out,training=training)
    def get_config(self):
        return {"image_size":self.image_size,"patch_size":self.patch_size,"num_patches":self.num_patches,"D":self.D,"num_heads":self.num_heads,"num_layers":self.num_layers,"mlp_dim":self.mlp_dim,"dropout":self.dropout,"vocab_size":self.vocab_size,"max_len":self.max_len,"batch_size":self.batch_size,"epochs":self.epochs,"save_path":self.save_path,"max_len_input":self.max_len_input,"learning_rate":self.learning_rate}

class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def call(self,y_true,y_pred):
        mask=tf.cast(tf.not_equal(y_true,0),tf.float32)
        loss=keras.losses.sparse_categorical_crossentropy(y_true,y_pred,from_logits=True)
        return tf.reduce_sum(loss*mask)



def load_db():
    defaults={"api_keys":{},"inf_log":[],"samples":[],"stats":{"inferences":0,"train_steps":0,"avg_gen_time":0.0,"total_gpu_time":0.0},"config":{"model_name":MODEL_NAME,"version":MODEL_VERSION}}
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            d=json.load(f)
        # Ensure all top-level keys exist (handles old/mismatched schemas)
        for k,v in defaults.items():
            if k not in d:d[k]=v
        if "stats" in d:
            for sk,sv in defaults["stats"].items():
                if sk not in d["stats"]:d["stats"][sk]=sv
        return d
    return defaults

def save_db(d):
    with open(DB_PATH,"w") as f:json.dump(d,f,indent=2)

def get_system_info():
    try:
        cpu_pct=psutil.cpu_percent(interval=0.1)
        mem=psutil.virtual_memory()
        disk=psutil.disk_usage(BASE)
        return{"cpu_percent":cpu_pct,"mem_percent":mem.percent,"mem_used_gb":round(mem.used/1024**3,2),"mem_total_gb":round(mem.total/1024**3,2),"disk_used_gb":round(disk.used/1024**3,2)}
    except:return{"cpu_percent":0,"mem_percent":0,"mem_used_gb":0,"mem_total_gb":0,"disk_used_gb":0}

def get_gpu_stats():
    stats={"util_pct":0,"mem_used_mb":0,"mem_total_mb":0,"mem_pct":0,"available":gpu_info.get("avail",False),"name":gpu_info.get("name","CPU")}
    if gpu_info.get("avail"):
        try:
            import pynvml;pynvml.nvmlInit()
            h=pynvml.nvmlDeviceGetHandleByIndex(0)
            u=pynvml.nvmlDeviceGetUtilizationRates(h)
            m=pynvml.nvmlDeviceGetMemoryInfo(h)
            stats.update({"util_pct":u.gpu,"mem_used_mb":round(m.used/1024**2),"mem_total_mb":round(m.total/1024**2),"mem_pct":round(m.used/m.total*100,1)})
            return stats
        except:pass
        try:
            info=tf.config.experimental.get_memory_info('GPU:0')
            cur=round(info.get("current",0)/1024**2)
            pk=round(info.get("peak",0)/1024**2)
            stats.update({"mem_used_mb":cur,"mem_total_mb":pk or cur})
            if pk>0:stats["mem_pct"]=round(cur/pk*100,1)
        except:pass
    else:
        try:
            info=tf.config.experimental.get_memory_info('CPU:0')
            stats["mem_used_mb"]=round(info.get("current",0)/1024**2)
        except:pass
    return stats

def load_tok():
    global vocab,id2tok,merges
    with open(VOCAB_PATH) as f:vocab=json.load(f)
    id2tok={v:k for k,v in vocab.items()}
    with open(MERGES_PATH) as f:lines=f.read().splitlines()
    merges=[tuple(l.split()) for l in lines if l and not l.startswith("#") and len(l.split())==2]

def get_pairs(w):
    p=set();prev=w[0]
    for c in w[1:]:p.add((prev,c));prev=c
    return p

def bpe(token):
    w=tuple(token);ps=get_pairs(w)
    if not ps:return w
    rk={m:i for i,m in enumerate(merges)}
    while True:
        v={p for p in ps if p in rk}
        if not v:break
        bg=min(v,key=lambda p:rk[p]);a,b=bg;nw=[];i=0
        while i<len(w):
            try:
                j=w.index(a,i);nw.extend(w[i:j])
                if j<len(w)-1 and w[j+1]==b:nw.append(a+b);i=j+2
                else:nw.append(w[j]);i=j+1
            except ValueError:nw.extend(w[i:]);break
        w=tuple(nw);ps=get_pairs(w)
    return w

def tokenize(text):
    text=text.lower().strip();toks=[]
    for i,word in enumerate(text.split()):
        for j,piece in enumerate(bpe(word)):
            tok=("\u0120"+piece) if(i>0 and j==0) else piece
            toks.append(tok)
    return[vocab.get(t,UNK) for t in toks]

def detokenize(ids):
    toks=[id2tok.get(i,"") for i in ids if i not in(PAD,ST,ED)]
    return "".join(toks).replace("\u0120"," ").strip()

def preprocess(img):
    img=img.convert("RGB").resize((SZ,SZ),Image.LANCZOS)
    a=np.array(img,dtype=np.float32)/255.0
    a=(a-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
    # Extract 16x16 patches: (224,224,3) → (196,768) then add batch dim
    # tf.image.extract_patches: (1,224,224,3) → (1,14,14,768) → (1,196,768)
    t=tf.constant(a[np.newaxis],dtype=tf.float32)
    p=tf.image.extract_patches(t,sizes=[1,16,16,1],strides=[1,16,16,1],rates=[1,1,1,1],padding='VALID')
    return p.numpy().reshape(1,196,768)

def gen_caption(img,temp=1.0,beam=1):
    t0=time.time();ia=preprocess(img)
    # MX-1: leave one slot so the logit index (len(tks)-1) never exceeds MX-1=39
    MAX_NEW=MX-1
    if beam<=1:
        tks=[ST]
        for _ in range(MAX_NEW):
            # Pad/clip to exactly MX tokens so the model always sees the right shape
            padded=tks[:MX]+[PAD]*(MX-len(tks[:MX]))
            sq=np.array([padded],dtype=np.int32)
            lg=model.predict([ia,sq],verbose=0)
            idx=min(len(tks)-1,MX-1)  # clamp index so it's always within [0, MX-1]
            nl=lg[0,idx,:]
            if temp!=1.0:nl=nl/temp
            nid=int(np.argmax(nl))
            if nid==ED:break
            tks.append(nid)
        cap=detokenize(tks[1:])
    else:
        bms=[(0.0,[ST])]
        for _ in range(MAX_NEW):
            nb=[]
            for sc,tk in bms:
                if tk[-1]==ED:nb.append((sc,tk));continue
                padded=tk[:MX]+[PAD]*(MX-len(tk[:MX]))
                sq=np.array([padded],dtype=np.int32)
                lg=model.predict([ia,sq],verbose=0)
                idx=min(len(tk)-1,MX-1)
                pr=tf.nn.softmax(lg[0,idx,:]).numpy()
                for tid in np.argsort(pr)[-beam:]:
                    nb.append((sc+np.log(pr[tid]+1e-9),tk+[int(tid)]))
            bms=sorted(nb,key=lambda x:x[0],reverse=True)[:beam]
            if all(b[1][-1]==ED for b in bms):break
        cap=detokenize(bms[0][1][1:])
    dt=round(time.time()-t0,3)
    sys_info=get_system_info()
    return{"caption":cap,"time_sec":dt,"gpu":gpu_info["name"],"device":"GPU" if gpu_info["avail"] else "CPU","cpu_usage":sys_info["cpu_percent"],"mem_usage":sys_info["mem_percent"]}

def get_optimizer():
    global _optimizer
    if _optimizer is None:
        _optimizer=keras.optimizers.Adam(learning_rate=1e-5)
    return _optimizer

def finetune(img,caption_text):
    ia=preprocess(img)
    ids=[ST]+tokenize(caption_text)+[ED]
    ids=ids[:MX]+[PAD]*(MX-len(ids[:MX]))
    ia2=np.array([ids],dtype=np.int32)
    tgt=np.roll(ia2,-1,axis=1);tgt[0,-1]=PAD
    with tf.GradientTape() as tape:
        lg=model([ia,ia2],training=True)
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True,ignore_class=PAD)(tgt,lg)
    gr=tape.gradient(loss,model.trainable_variables)
    get_optimizer().apply_gradients(zip(gr,model.trainable_variables))
    return float(loss)

def req_admin(f):
    @wraps(f)
    def d(*a,**k):
        if request.headers.get("X-Admin-Token")!="neurallens-admin-2026":
            return jsonify({"error":"Admin required"}),403
        return f(*a,**k)
    return d

def req_key(f):
    @wraps(f)
    def d(*a,**k):
        key=request.headers.get("X-API-Key") or request.args.get("api_key")
        if not key:return jsonify({"error":"API key required"}),401
        db=load_db()
        if key not in db["api_keys"]:return jsonify({"error":"Invalid key"}),403
        e=db["api_keys"][key]
        if e.get("expires") and datetime.fromisoformat(e["expires"])<datetime.utcnow():
            return jsonify({"error":"Key expired"}),403
        e["calls"]=e.get("calls",0)+1;save_db(db)
        return f(*a,**k)
    return d

app=Flask(__name__,static_folder=None)
app.secret_key=secrets.token_hex(16)
CORS(app,resources={r"/api/*":{"origins":"*"}})

@app.route("/")
def index():return send_from_directory(BASE,"index.html")


@app.route('/favicon.ico')
def favicon():
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
    <rect width="100%" height="100%" fill="#111827"/>
    <text x="50%" y="55%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="28" fill="#f59e0b">VM</text>
</svg>'''
        return Response(svg, mimetype='image/svg+xml')

@app.route("/api/health")
def health():
    sys_info=get_system_info()
    return jsonify({"status":"ok","model_loaded":model is not None,"gpu":gpu_info,"system":sys_info,"model_name":MODEL_NAME,"version":MODEL_VERSION})

@app.route("/api/v1/health")
def health_v1():return health()

@app.route("/api/model/info")
def model_info():
    return jsonify({"name":MODEL_NAME,"version":MODEL_VERSION,"description":MODEL_DESC,"model_size_gb":1.3,"architecture":"ViT+GPT-2","max_caption_length":40,"input_size":224,"gpu_available":gpu_info.get("avail",False),"device":gpu_info.get("name","CPU")})

@app.route("/api/caption",methods=["POST"])
def caption():
    if model is None:return jsonify({"error":"Model loading…"}),503
    try:
        if "image" in request.files:
            img=Image.open(request.files["image"].stream)
            temp=float(request.form.get("temperature","1.0"))
            beam=int(request.form.get("beam","1"))
        else:
            d=request.get_json();img=Image.open(io.BytesIO(base64.b64decode(d["image_b64"])))
            temp=float(d.get("temperature",1.0));beam=int(d.get("beam",1))
        r=gen_caption(img,temp,beam)
        db=load_db()
        if "avg_gen_time" not in db["stats"]:db["stats"]["avg_gen_time"]=0.0
        old_avg=db["stats"].get("avg_gen_time",0.0)
        n=db["stats"].get("inferences",0)
        db["stats"]["avg_gen_time"]=round((old_avg*n+r["time_sec"])/(n+1),4)
        db["stats"]["inferences"]=n+1
        db["inf_log"].append({"id":uuid.uuid4().hex[:8],"cap":r["caption"],"t":r["time_sec"],"device":r["device"],"ts":datetime.utcnow().isoformat()})
        db["inf_log"]=db["inf_log"][-200:];save_db(db)
        return jsonify(r)
    except Exception as e:return jsonify({"error":str(e),"trace":traceback.format_exc()}),500

@app.route("/api/v1/caption",methods=["POST"])
@req_key
def caption_api():return caption()

@app.route("/api/train/upload",methods=["POST"])
def train_upload():
    try:
        if "image" not in request.files:return jsonify({"error":"No image"}),400
        cap=request.form.get("caption","").strip()
        if not cap:return jsonify({"error":"No caption"}),400
        img=Image.open(request.files["image"].stream);sid=uuid.uuid4().hex[:8]
        ip=os.path.join(SAMPLES,f"{sid}.jpg");img.convert("RGB").save(ip,"JPEG")
        db=load_db()
        if "samples" not in db: db["samples"] = []
        db["samples"].append({"id":sid,"caption":cap,"img_path":ip,"added":datetime.utcnow().isoformat(),"trained":False})
        save_db(db);return jsonify({"sample_id":sid,"caption":cap,"status":"queued"})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/train/run/<sid>",methods=["POST"])
def train_run(sid):
    if model is None:return jsonify({"error":"Model not loaded"}),503
    db=load_db();s=next((x for x in db["samples"] if x["id"]==sid),None)
    if not s:return jsonify({"error":"Not found"}),404
    body=request.get_json(silent=True) or {}
    epochs=max(1,min(int(body.get("epochs",1)),50))
    def _t():
        global tstatus
        tstatus={"active":True,"progress":10,"msg":"Loading…","log":[]}
        try:
            img=Image.open(s["img_path"]);losses=[]
            for ep in range(epochs):
                tstatus["progress"]=10+int(ep/epochs*80)
                tstatus["msg"]=f"Epoch {ep+1}/{epochs}…"
                loss=finetune(img,s["caption"]);losses.append(loss)
                tstatus["log"].append(f"Epoch {ep+1}: loss={loss:.4f}")
            final_loss=losses[-1];dt=round(time.time(),3)
            d2=load_db()
            for x in d2["samples"]:
                if x["id"]==sid:x["trained"]=True;x["loss"]=final_loss;x["epochs"]=epochs
            d2["stats"]["train_steps"]+=epochs;save_db(d2)
            tstatus={"active":False,"progress":100,"msg":f"Done! {epochs} epochs, final loss={final_loss:.4f}","log":tstatus["log"]}
        except Exception as e:
            tstatus={"active":False,"progress":0,"msg":f"Error: {e}","log":[traceback.format_exc()]}
    threading.Thread(target=_t,daemon=True).start()
    return jsonify({"status":"started"})


@app.route("/api/train/run_all",methods=["POST"])
def train_all():
    if model is None:return jsonify({"error":"Model not loaded"}),503
    body=request.get_json(silent=True) or {}
    epochs=max(1,min(int(body.get("epochs",1)),50))
    def _ta():
        global tstatus
        db=load_db();un=[s for s in db["samples"] if not s.get("trained")];tot=len(un)
        if not tot:tstatus={"active":False,"progress":100,"msg":"Nothing to train.","log":[]};return
        tstatus={"active":True,"progress":0,"msg":f"Training {tot} samples × {epochs} epochs…","log":[]}
        for i,s in enumerate(un):
            try:
                img=Image.open(s["img_path"]);losses=[]
                for ep in range(epochs):
                    loss=finetune(img,s["caption"]);losses.append(loss)
                    tstatus["log"].append(f"{s['id']} ep{ep+1} loss={loss:.4f}")
                final_loss=losses[-1]
                d2=load_db()
                for x in d2["samples"]:
                    if x["id"]==s["id"]:x["trained"]=True;x["loss"]=final_loss;x["epochs"]=epochs
                d2["stats"]["train_steps"]+=epochs;save_db(d2)
                tstatus["progress"]=int((i+1)/tot*100)
                tstatus["msg"]=f"Sample {i+1}/{tot} — loss={final_loss:.4f}"
            except Exception as e:tstatus["log"].append(f"ERR {s['id']}: {e}")
        tstatus["active"]=False;tstatus["progress"]=100;tstatus["msg"]=f"All {tot} samples done!"
    threading.Thread(target=_ta,daemon=True).start()
    return jsonify({"status":"started"})


@app.route("/api/train/status")
def train_st():return jsonify(tstatus)

@app.route("/api/train/samples")
def train_samples():return jsonify(load_db().get("samples",[]))

@app.route("/api/train/samples/<sid>",methods=["DELETE"])
def del_sample(sid):
    db=load_db();s=next((x for x in db["samples"] if x["id"]==sid),None)
    if not s:return jsonify({"error":"Not found"}),404
    db["samples"]=[x for x in db["samples"] if x["id"]!=sid];save_db(db)
    try:os.remove(s["img_path"])
    except:pass
    return jsonify({"status":"deleted"})

@app.route("/api/train/samples/<sid>/thumb")
def thumb(sid):
    db=load_db();s=next((x for x in db["samples"] if x["id"]==sid),None)
    if not s or not os.path.exists(s["img_path"]):return jsonify({"error":"Not found"}),404
    img=Image.open(s["img_path"]).convert("RGB");img.thumbnail((120,120))
    buf=io.BytesIO();img.save(buf,"JPEG");b=base64.b64encode(buf.getvalue()).decode()
    return jsonify({"thumb":f"data:image/jpeg;base64,{b}"})

@app.route("/api/keys/create",methods=["POST"])
@req_admin
def create_key():
    d=request.get_json() or {};label=d.get("label","key");days=int(d.get("days",30))
    key="nlk-"+secrets.token_urlsafe(32);db=load_db()
    db["api_keys"][key]={"label":label,"calls":0,"created":datetime.utcnow().isoformat(),"expires":(datetime.utcnow()+timedelta(days=days)).isoformat() if days>0 else None}
    save_db(db);return jsonify({"key":key,"label":label})

@app.route("/api/keys/list")
@req_admin
def list_keys():return jsonify(load_db().get("api_keys",{}))

@app.route("/api/keys/<key>",methods=["DELETE"])
@req_admin
def del_key(key):
    db=load_db()
    if key not in db["api_keys"]:return jsonify({"error":"Not found"}),404
    del db["api_keys"][key];save_db(db);return jsonify({"status":"deleted"})

@app.route("/api/caption/batch",methods=["POST"])
@req_key
def batch_caption():
    """Advanced batch processing with multiple images"""
    if model is None:return jsonify({"error":"Model not loaded"}),503
    try:
        images=request.files.getlist("images")
        if not images or len(images)>10:
            return jsonify({"error":"Upload 1-10 images"}),400
        
        results=[]
        for img_file in images:
            img=Image.open(img_file.stream)
            temp=float(request.form.get("temperature","1.0"))
            beam=int(request.form.get("beam","1"))
            r=gen_caption(img,temp,beam)
            results.append(r)
        
        return jsonify({"count":len(results),"results":results})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/caption/export/<format>",methods=["GET"])
@req_key
def export_captions(format):
    """Export inference history in different formats"""
    db=load_db()
    logs=db.get("inf_log",[])
    
    if format=="json":
        return jsonify(logs)
    elif format=="csv":
        csv_lines=["ID,Caption,Time,Device,Timestamp"]
        for log in logs:
            csv_lines.append(f"{log.get('id','')},\"{log.get('cap','')}\",{log.get('t',0)},{log.get('device','')},{log.get('ts','')}")
        resp=jsonify("\n".join(csv_lines))
        resp.headers["Content-Disposition"]="attachment;filename=captions_export.csv"
        return resp
    elif format=="txt":
        txt_lines=[f"[{log.get('ts','')}] {log.get('cap','')}" for log in logs]
        resp=jsonify("\n\n".join(txt_lines))
        resp.headers["Content-Disposition"]="attachment;filename=captions_export.txt"
        return resp
    else:
        return jsonify({"error":"Unsupported format"}),400

@app.route("/api/train/advanced",methods=["POST"])
def train_advanced():
    """Advanced training with custom learning rate and epochs"""
    if model is None:return jsonify({"error":"Model not loaded"}),503
    try:
        d=request.get_json()
        sid=d.get("sample_id")
        lr=float(d.get("learning_rate",1e-5))
        epochs=int(d.get("epochs",1))
        
        db=load_db()
        s=next((x for x in db["samples"] if x["id"]==sid),None)
        if not s:return jsonify({"error":"Sample not found"}),404
        
        def _advanced_train():
            global tstatus
            tstatus={"active":True,"progress":0,"msg":"Advanced training…","log":[]}
            try:
                img=Image.open(s["img_path"])
                losses=[]
                for epoch in range(epochs):
                    loss=finetune(img,s["caption"])
                    losses.append(float(loss))
                    tstatus["progress"]=int((epoch+1)/epochs*100)
                    tstatus["msg"]=f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}"
                    tstatus["log"].append(f"Epoch {epoch+1}: {loss:.4f}")
                
                d2=load_db()
                for x in d2["samples"]:
                    if x["id"]==sid:
                        x["trained"]=True
                        x["loss"]=losses[-1]
                        x["history"]=losses
                d2["stats"]["train_steps"]+=epochs
                save_db(d2)
                tstatus["active"]=False
                tstatus["msg"]=f"Done! Final loss: {losses[-1]:.4f}"
            except Exception as e:
                tstatus={"active":False,"progress":0,"msg":f"Error: {e}","log":[]}
        
        threading.Thread(target=_advanced_train,daemon=True).start()
        return jsonify({"status":"started"})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/stats")
def stats():
    db=load_db();sys_info=get_system_info();gpu_stats=get_gpu_stats()
    return jsonify({"stats":db.get("stats",{}),"recent":db.get("inf_log",[])[-10:],"samples":len(db.get("samples",[])),"keys":len(db.get("api_keys",{})),"gpu":gpu_info,"gpu_stats":gpu_stats,"system":sys_info,"model":{"name":MODEL_NAME,"version":MODEL_VERSION}})

@app.route("/api/telemetry")
def telemetry():
    s=get_system_info();g=get_gpu_stats()
    return jsonify({"cpu":s["cpu_percent"],"ram":s["mem_percent"],"ram_used_gb":s["mem_used_gb"],"ram_total_gb":s["mem_total_gb"],"disk_gb":s["disk_used_gb"],"gpu":g,"model_loaded":model is not None,"ts":datetime.utcnow().isoformat()})

# Extra models removed as requested.


if __name__=="__main__":
    print("\n" + "="*60)
    print(f"🚀 {MODEL_NAME} {MODEL_VERSION}")
    print(f"📝 {MODEL_DESC}")
    print("="*60)
    print("[1/4] Loading tokenizer…")
    load_tok()
    print(f"[2/4] Loading model ({1.3}GB)…")
    custom_objects={"ImageCaptioningModel":ImageCaptioningModel,"MaskedSparseCategoricalCrossentropy":MaskedSparseCategoricalCrossentropy,"_ViTEncoder":_ViTEncoder,"_GPTDecoder":_GPTDecoder}
    model=keras.models.load_model(MODEL_PATH,custom_objects=custom_objects,compile=False)
    print(f"  ✓ Weights loaded: {len(model.weights)} tensors")
    print("[3/4] Detecting GPU…")
    gpus=tf.config.list_physical_devices("GPU")
    gpu_info={"name":gpus[0].name,"avail":True} if gpus else {"name":"CPU","avail":False}
    print(f"[4/4] System check…")
    sys_info=get_system_info()
    print("="*60)
    print(f"✓ Processing Device: {gpu_info['name']}")
    print(f"✓ CPU Usage: {sys_info['cpu_percent']}%")
    print(f"✓ Memory: {sys_info['mem_percent']}%")
    print(f"✓ Web Server:     http://localhost:5055")
    print(f"✓ AI Studio:      http://localhost:5055")
    print(f"✓ Caption API:    http://localhost:5055/api/caption")
    print("="*60)
    app.run(host="0.0.0.0",port=5055,debug=False,threaded=True)

