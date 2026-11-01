//! Lossless storage rewrite for reg words whose nibble codes are0..7.
const std=@import("std");
pub fn pack(wide:[]const u32,out:[]u32)!void{
 if(wide.len%256!=0 or out.len!=wide.len/4*3)return error.InvalidStorage;
 for(wide)|word|if(word&0x88888888!=0)return error.WideCodes;
 for(0..wide.len/256)|block|for(0..32)|lane|{
  const src=wide[block*256..][0..256];const dst=out[block*192..][0..192];
  for(0..4)|w|dst[lane*4+w]=(src[lane*4+w]&0x33333333)|((src[128+lane*4+w]&0x33333333)<<2);
  for(0..2)|half|{
   var bits:u32=0;for(0..4)|w|{for(0..8)|j|{bits|=((src[half*128+lane*4+w]>>@as(u5,@intCast(4*j+2)))&1)<<@as(u5,@intCast(8*w+j));}}
   dst[128+lane*2+half]=bits;
  }
 };
}
pub fn unpack(compact:[]const u32,out:[]u32)!void{
 if(compact.len%192!=0 or out.len!=compact.len/3*4)return error.InvalidStorage;
 for(0..compact.len/192)|block|for(0..32)|lane|for(0..8)|w|{
  const src=compact[block*192..][0..192];const low=src[lane*4+w%4];
  var word:u32=if(w<4)low&0x33333333 else(low>>2)&0x33333333;
  const high=src[128+lane*2+w/4]>>@as(u5,@intCast(8*(w%4)));
  for(0..8)|j|word|=((high>>@as(u5,@intCast(j)))&1)<<@as(u5,@intCast(4*j+2));
  out[block*256+(w/4)*128+lane*4+w%4]=word;
 };
}
