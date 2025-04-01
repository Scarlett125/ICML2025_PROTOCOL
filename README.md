## Environment Setup

```bash
step1: conda create -n protocol_test python=3.8.18
step2: conda activate protocol_test
step3: pip install -r requirements.txt
step4: pip install protocol_lib-0.1-*.whl
```

step5: Modify transformer.py and activation.py located in the 'protocol_test' virtual environment
  （1）In activation.py: Find two instances of need_weights=need_weights; Modify both to need_weights=True.
  （2）In transformer.py: Locate code1; Replace it with code2.
 Code1: 
        -------------------------------------------------------------------------------------
                x = src
                if self.norm_first:
                    x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
                    x = x + self._ff_block(self.norm2(x))
                else:
                    x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
                    x = self.norm2(x + self._ff_block(x))

                return x

            # self-attention block
            def _sa_block(self, x: Tensor,
                        attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
                x = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False, is_causal=is_causal)[0]
                return self.dropout1(x)
        -------------------------------------------------------------------------------------
 Code2:
        -------------------------------------------------------------------------------------
                x = src
                first,  second = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                if self.norm_first:
                    x = x + first
                    x = x + self._ff_block(self.norm2(x))
                else:
                    x = self.norm1(x + first)
                    x = self.norm2(x + self._ff_block(x))

                return x,second

            # self-attention block
            def _sa_block(self, x: Tensor,
                        attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
                x,y = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
                return self.dropout1(x),y
        -------------------------------------------------------------------------------------

## Running Test
```bash
python test.py
```

## Running Train
```bash
python train.py
```