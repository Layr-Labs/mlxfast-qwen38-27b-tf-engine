// The drafter selector of a depth-one block on the GPU (tf_draft_select), on exact binary64 words.
#include <metal_stdlib>
using namespace metal;
constant ulong tf_log_poly1[11] = {0xbfe0000000000000ul, 0x3fd5555555555577ul, 0xbfcffffffffffdcbul, 0x3fc999999995dd0cul, 0xbfc55555556745a7ul, 0x3fc24924a344de30ul, 0xbfbfffffa4423d65ul, 0x3fbc7184282ad6caul, 0xbfb999eb43b068fful, 0x3fb78182f7afd085ul, 0xbfb5521375d145cdul};
constant ulong tf_log_poly[5] = {0xbfe0000000000001ul, 0x3fd555555551305bul, 0xbfcfffffffeb4590ul, 0x3fc999b324f10111ul, 0xbfc55575e506c89ful};
constant ulong2 tf_log_tab[128] = {ulong2(0x3ff734f0c3e0de9ful, 0xbfd7cc7f79e69000ul), ulong2(0x3ff713786a2ce91ful, 0xbfd76feec20d0000ul), ulong2(0x3ff6f26008fab5a0ul, 0xbfd713e31351e000ul), ulong2(0x3ff6d1a61f138c7dul, 0xbfd6b85b38287800ul), ulong2(0x3ff6b1490bc5b4d1ul, 0xbfd65d5590807800ul), ulong2(0x3ff69147332f0cbaul, 0xbfd602d076180000ul), ulong2(0x3ff6719f18224223ul, 0xbfd5a8ca86909000ul), ulong2(0x3ff6524f99a51ed9ul, 0xbfd54f4356035000ul), ulong2(0x3ff63356aa8f24c4ul, 0xbfd4f637c36b4000ul), ulong2(0x3ff614b36b9ddc14ul, 0xbfd49da7fda85000ul), ulong2(0x3ff5f66452c65c4cul, 0xbfd445923989a800ul), ulong2(0x3ff5d867b5912c4ful, 0xbfd3edf439b0b800ul), ulong2(0x3ff5babccb5b90deul, 0xbfd396ce448f7000ul), ulong2(0x3ff59d61f2d91a78ul, 0xbfd3401e17bda000ul), ulong2(0x3ff5805612465687ul, 0xbfd2e9e2ef468000ul), ulong2(0x3ff56397cee76bd3ul, 0xbfd2941b3830e000ul), ulong2(0x3ff54725e2a77f93ul, 0xbfd23ec58cda8800ul), ulong2(0x3ff52aff42064583ul, 0xbfd1e9e129279000ul), ulong2(0x3ff50f22dbb2bddful, 0xbfd1956d2b48f800ul), ulong2(0x3ff4f38f4734ded7ul, 0xbfd141679ab9f800ul), ulong2(0x3ff4d843cfde2840ul, 0xbfd0edd094ef9800ul), ulong2(0x3ff4bd3ec078a3c8ul, 0xbfd09aa518db1000ul), ulong2(0x3ff4a27fc3e0258aul, 0xbfd047e65263b800ul), ulong2(0x3ff4880524d48434ul, 0xbfcfeb224586f000ul), ulong2(0x3ff46dce1b192d0bul, 0xbfcf474a7517b000ul), ulong2(0x3ff453d9d3391854ul, 0xbfcea4443d103000ul), ulong2(0x3ff43a2744b4845aul, 0xbfce020d44e9b000ul), ulong2(0x3ff420b54115f8fbul, 0xbfcd60a22977f000ul), ulong2(0x3ff40782da3ef4b1ul, 0xbfccc00104959000ul), ulong2(0x3ff3ee8f5d57fe8ful, 0xbfcc202956891000ul), ulong2(0x3ff3d5d9a00b4ce9ul, 0xbfcb81178d811000ul), ulong2(0x3ff3bd60c010c12bul, 0xbfcae2c9ccd3d000ul), ulong2(0x3ff3a5242b75dab8ul, 0xbfca45402e129000ul), ulong2(0x3ff38d22cd9fd002ul, 0xbfc9a877681df000ul), ulong2(0x3ff3755bc5847a1cul, 0xbfc90c6d69483000ul), ulong2(0x3ff35dce49ad36e2ul, 0xbfc87120a645c000ul), ulong2(0x3ff34679984dd440ul, 0xbfc7d68fb4143000ul), ulong2(0x3ff32f5cceffcb24ul, 0xbfc73cb83c627000ul), ulong2(0x3ff3187775a10d49ul, 0xbfc6a39a9b376000ul), ulong2(0x3ff301c8373e3990ul, 0xbfc60b3154b7a000ul), ulong2(0x3ff2eb4ebb95f841ul, 0xbfc5737d76243000ul), ulong2(0x3ff2d50a0219a9d1ul, 0xbfc4dc7b8fc23000ul), ulong2(0x3ff2bef9a8b7fd2aul, 0xbfc4462c51d20000ul), ulong2(0x3ff2a91c7a0c1babul, 0xbfc3b08abc830000ul), ulong2(0x3ff293726014b530ul, 0xbfc31b996b490000ul), ulong2(0x3ff27dfa5757a1f5ul, 0xbfc2875490a44000ul), ulong2(0x3ff268b39b1d3bbful, 0xbfc1f3b9f879a000ul), ulong2(0x3ff2539d838ff5bdul, 0xbfc160c8252ca000ul), ulong2(0x3ff23eb7aac9083bul, 0xbfc0ce7f57f72000ul), ulong2(0x3ff22a012ba940b6ul, 0xbfc03cdc49fea000ul), ulong2(0x3ff2157996cc4132ul, 0xbfbf57bdbc4b8000ul), ulong2(0x3ff201201dd2fc9bul, 0xbfbe370896404000ul), ulong2(0x3ff1ecf4494d480bul, 0xbfbd17983ef94000ul), ulong2(0x3ff1d8f5528f6569ul, 0xbfbbf9674ed8a000ul), ulong2(0x3ff1c52311577e7cul, 0xbfbadc79202f6000ul), ulong2(0x3ff1b17c74cb26e9ul, 0xbfb9c0c3e7288000ul), ulong2(0x3ff19e010c2c1ab6ul, 0xbfb8a646b372c000ul), ulong2(0x3ff18ab07bb670bdul, 0xbfb78d01b3ac0000ul), ulong2(0x3ff1778a25efbcb6ul, 0xbfb674f145380000ul), ulong2(0x3ff1648d354c31daul, 0xbfb55e0e6d878000ul), ulong2(0x3ff151b990275fddul, 0xbfb4485cdea1e000ul), ulong2(0x3ff13f0ea432d24cul, 0xbfb333d94d6aa000ul), ulong2(0x3ff12c8b7210f9daul, 0xbfb22079f8c56000ul), ulong2(0x3ff11a3028ecb531ul, 0xbfb10e4698622000ul), ulong2(0x3ff107fbda8434aful, 0xbfaffa6c6ad20000ul), ulong2(0x3ff0f5ee0f4e6bb3ul, 0xbfadda8d4a774000ul), ulong2(0x3ff0e4065d2a9fceul, 0xbfabbcece4850000ul), ulong2(0x3ff0d244632ca521ul, 0xbfa9a1894012c000ul), ulong2(0x3ff0c0a77ce2981aul, 0xbfa788583302c000ul), ulong2(0x3ff0af2f83c636d1ul, 0xbfa5715e67d68000ul), ulong2(0x3ff09ddb98a01339ul, 0xbfa35c8a49658000ul), ulong2(0x3ff08cabaf52e7dful, 0xbfa149e364154000ul), ulong2(0x3ff07b9f2f4e28fbul, 0xbf9e72c082eb8000ul), ulong2(0x3ff06ab58c358f19ul, 0xbf9a55f152528000ul), ulong2(0x3ff059eea5ecf92cul, 0xbf963d62cf818000ul), ulong2(0x3ff04949cdd12c90ul, 0xbf9228fb8caa0000ul), ulong2(0x3ff038c6c6f0ada9ul, 0xbf8c317b20f90000ul), ulong2(0x3ff02865137932a9ul, 0xbf8419355daa0000ul), ulong2(0x3ff0182427ea7348ul, 0xbf781203c2ec0000ul), ulong2(0x3ff008040614b195ul, 0xbf60040979240000ul), ulong2(0x3fefe01ff726fa1aul, 0x3f6feff384900000ul), ulong2(0x3fefa11cc261ea74ul, 0x3f87dc41353d0000ul), ulong2(0x3fef6310b081992eul, 0x3f93cea3c4c28000ul), ulong2(0x3fef25f63ceeadcdul, 0x3f9b9fc114890000ul), ulong2(0x3feee9c8039113e7ul, 0x3fa1b0d8ce110000ul), ulong2(0x3feeae8078cbb1abul, 0x3fa58a5bd001c000ul), ulong2(0x3fee741aa29d0c9bul, 0x3fa95c8340d88000ul), ulong2(0x3fee3a91830a99b5ul, 0x3fad276aef578000ul), ulong2(0x3fee01e009609a56ul, 0x3fb07598e598c000ul), ulong2(0x3fedca01e577bb98ul, 0x3fb253f5e30d2000ul), ulong2(0x3fed92f20b7c9103ul, 0x3fb42edd8b380000ul), ulong2(0x3fed5cac66fb5cceul, 0x3fb606598757c000ul), ulong2(0x3fed272caa5ede9dul, 0x3fb7da76356a0000ul), ulong2(0x3fecf26e3e6b2ccdul, 0x3fb9ab434e1c6000ul), ulong2(0x3fecbe6da2a77902ul, 0x3fbb78c7bb0d6000ul), ulong2(0x3fec8b266d37086dul, 0x3fbd431332e72000ul), ulong2(0x3fec5894bd5d5804ul, 0x3fbf0a3171de6000ul), ulong2(0x3fec26b533bb9f8cul, 0x3fc067152b914000ul), ulong2(0x3febf583eeece73ful, 0x3fc147858292b000ul), ulong2(0x3febc4fd75db96c1ul, 0x3fc2266ecdca3000ul), ulong2(0x3feb951e0c864a28ul, 0x3fc303d7a6c55000ul), ulong2(0x3feb65e2c5ef3e2cul, 0x3fc3dfc33c331000ul), ulong2(0x3feb374867c9888bul, 0x3fc4ba366b7a8000ul), ulong2(0x3feb094b211d304aul, 0x3fc5933928d1f000ul), ulong2(0x3feadbe885f2ef7eul, 0x3fc66acd2418f000ul), ulong2(0x3feaaf1d31603da2ul, 0x3fc740f8ec669000ul), ulong2(0x3fea82e63fd358a7ul, 0x3fc815c0f51af000ul), ulong2(0x3fea5740ef09738bul, 0x3fc8e92954f68000ul), ulong2(0x3fea2c2a90ab4b27ul, 0x3fc9bb3602f84000ul), ulong2(0x3fea01a01393f2d1ul, 0x3fca8bed1c2c0000ul), ulong2(0x3fe9d79f24db3c1bul, 0x3fcb5b515c01d000ul), ulong2(0x3fe9ae2505c7b190ul, 0x3fcc2967ccbcc000ul), ulong2(0x3fe9852ef297ce2ful, 0x3fccf635d5486000ul), ulong2(0x3fe95cbaeea44b75ul, 0x3fcdc1bd3446c000ul), ulong2(0x3fe934c69de74838ul, 0x3fce8c01b8cfe000ul), ulong2(0x3fe90d4f2f6752e6ul, 0x3fcf5509c0179000ul), ulong2(0x3fe8e6528effd79dul, 0x3fd00e6c121fb800ul), ulong2(0x3fe8bfce9fcc007cul, 0x3fd071b80e93d000ul), ulong2(0x3fe899c0dabec30eul, 0x3fd0d46b9e867000ul), ulong2(0x3fe87427aa2317fbul, 0x3fd13687334bd000ul), ulong2(0x3fe84f00acb39a08ul, 0x3fd1980d67234800ul), ulong2(0x3fe82a49e8653e55ul, 0x3fd1f8ffe0cc8000ul), ulong2(0x3fe8060195f40260ul, 0x3fd2595fd7636800ul), ulong2(0x3fe7e22563e0a329ul, 0x3fd2b9300914a800ul), ulong2(0x3fe7beb377dcb5adul, 0x3fd3187210436000ul), ulong2(0x3fe79baa679725c2ul, 0x3fd377266dec1800ul), ulong2(0x3fe77907f2170657ul, 0x3fd3d54ffbaf3000ul), ulong2(0x3fe756cadbd6130cul, 0x3fd432eee32fe000ul)};
constant ulong2 tf_log_tab2[128] = {ulong2(0x3fe61000014fb66bul, 0x3c7e026c91425b3cul), ulong2(0x3fe63000034db495ul, 0x3c8dbfea48005d41ul), ulong2(0x3fe650000d94d478ul, 0x3c8e7fa786d6a5b7ul), ulong2(0x3fe67000074e6fadul, 0x3c61fcea6b54254cul), ulong2(0x3fe68ffffedf0faeul, 0xbc7c7e274c590efdul), ulong2(0x3fe6b0000763c5bcul, 0xbc8ac16848dcda01ul), ulong2(0x3fe6d0001e5cc1f6ul, 0x3c833f1c9d499311ul), ulong2(0x3fe6efffeb05f63eul, 0xbc7e80041ae22d53ul), ulong2(0x3fe710000e869780ul, 0x3c7bff6671097952ul), ulong2(0x3fe72ffffc67e912ul, 0x3c8c00e226bd8724ul), ulong2(0x3fe74fffdf81116aul, 0xbc6e02916ef101d2ul), ulong2(0x3fe770000f679c90ul, 0xbc67fc71cd549c74ul), ulong2(0x3fe78ffffa7ec835ul, 0x3c81bec19ef50483ul), ulong2(0x3fe7affffe20c2e6ul, 0xbc707e1729cc6465ul), ulong2(0x3fe7cfffed3fc900ul, 0xbc808072087b8b1cul), ulong2(0x3fe7efffe9261a76ul, 0x3c8dc0286d9df9aeul), ulong2(0x3fe81000049ca3e8ul, 0x3c897fd251e54c33ul), ulong2(0x3fe8300017932c8ful, 0xbc8afee9b630f381ul), ulong2(0x3fe850000633739cul, 0x3c89bfbf6b6535bcul), ulong2(0x3fe87000204289c6ul, 0xbc8bbf65f3117b75ul), ulong2(0x3fe88fffebf57904ul, 0xbc89006ea23dcb57ul), ulong2(0x3fe8b00022bc04dful, 0xbc7d00df38e04b0aul), ulong2(0x3fe8cfffe50c1b8aul, 0xbc88007146ff9f05ul), ulong2(0x3fe8effffc918e43ul, 0x3c83817bd07a7038ul), ulong2(0x3fe910001efa5fc7ul, 0x3c893e9176dfb403ul), ulong2(0x3fe9300013467bb9ul, 0x3c7f804e4b980276ul), ulong2(0x3fe94fffe6ee076ful, 0xbc8f7ef0d9ff622eul), ulong2(0x3fe96fffde3c12d1ul, 0xbc7082aa962638baul), ulong2(0x3fe98ffff4458a0dul, 0xbc87801b9164a8eful), ulong2(0x3fe9afffdd982e3eul, 0xbc8740e08a5a9337ul), ulong2(0x3fe9cfffed49fb66ul, 0x3c3fce08c19be000ul), ulong2(0x3fe9f00020f19c51ul, 0xbc8a3faa27885b0aul), ulong2(0x3fea10001145b006ul, 0x3c74ff489958da56ul), ulong2(0x3fea300007bbf6faul, 0x3c8cbeab8a2b6d18ul), ulong2(0x3fea500010971d79ul, 0x3c88fecadd787930ul), ulong2(0x3fea70001df52e48ul, 0xbc8f41763dd8abdbul), ulong2(0x3fea90001c593352ul, 0xbc8ebf0284c27612ul), ulong2(0x3feab0002a4f3e4bul, 0xbc69fd043cff3f5ful), ulong2(0x3feacfffd7ae1ed1ul, 0xbc823ee7129070b4ul), ulong2(0x3feaefffee510478ul, 0x3c6a063ee00edea3ul), ulong2(0x3feb0fffdb650d5bul, 0x3c5a06c8381f0ab9ul), ulong2(0x3feb2ffffeaaca57ul, 0xbc79011e74233c1dul), ulong2(0x3feb4fffd995badcul, 0xbc79ff1068862a9ful), ulong2(0x3feb7000249e659cul, 0x3c8aff45d0864f3eul), ulong2(0x3feb8ffff9871640ul, 0x3c7cfe7796c2c3f9ul), ulong2(0x3febafffd204cb4ful, 0xbc63ff27eef22bc4ul), ulong2(0x3febcfffd2415c45ul, 0xbc6cffb7ee3bea21ul), ulong2(0x3febeffff86309dful, 0xbc814103972e0b5cul), ulong2(0x3fec0fffe1b57653ul, 0x3c8bc16494b76a19ul), ulong2(0x3fec2ffff1fa57e3ul, 0xbc64feef8d30c6edul), ulong2(0x3fec4fffdcbfe424ul, 0xbc843f68bcec4775ul), ulong2(0x3fec6fffed54b9f7ul, 0x3c847ea3f053e0ecul), ulong2(0x3fec8fffeb998fd5ul, 0x3c7383068df992f1ul), ulong2(0x3fecb0002125219aul, 0xbc68fd8e64180e04ul), ulong2(0x3feccfffdd94469cul, 0x3c8e7ebe1cc7ea72ul), ulong2(0x3fecefffeafdc476ul, 0x3c8ebe39ad9f88feul), ulong2(0x3fed1000169af82bul, 0x3c757d91a8b95a71ul), ulong2(0x3fed30000d0ff71dul, 0x3c89c1906970c7daul), ulong2(0x3fed4fffea790fc4ul, 0xbc580e37c558fe0cul), ulong2(0x3fed70002edc87e5ul, 0xbc7f80d64dc10f44ul), ulong2(0x3fed900021dc82aaul, 0xbc747c8f94fd5c5cul), ulong2(0x3fedafffd86b0283ul, 0x3c8c7f1dc521617eul), ulong2(0x3fedd000296c4739ul, 0x3c88019eb2ffb153ul), ulong2(0x3fedefffe54490f5ul, 0x3c6e00d2c652cc89ul), ulong2(0x3fee0fffcdabf694ul, 0xbc7f8340202d69d2ul), ulong2(0x3fee2fffdb52c8ddul, 0x3c7b00c1ca1b0864ul), ulong2(0x3fee4ffff24216eful, 0x3c72ffa8b094ab51ul), ulong2(0x3fee6fffe88a5e11ul, 0xbc57f673b1efbe59ul), ulong2(0x3fee9000119eff0dul, 0xbc84808d5e0bc801ul), ulong2(0x3feeafffdfa51744ul, 0x3c780006d54320b5ul), ulong2(0x3feed0001a127fa1ul, 0xbc5002f860565c92ul), ulong2(0x3feef00007babcc4ul, 0xbc8540445d35e611ul), ulong2(0x3fef0ffff57a8d02ul, 0xbc4ffb3139ef9105ul), ulong2(0x3fef30001ee58ac7ul, 0x3c8a81acf2731155ul), ulong2(0x3fef4ffff5823494ul, 0x3c8a3f41d4d7c743ul), ulong2(0x3fef6ffffca94c6bul, 0xbc6202f41c987875ul), ulong2(0x3fef8fffe1f9c441ul, 0x3c777dd1f477e74bul), ulong2(0x3fefafffd2e0e37eul, 0xbc6f01199a7ca331ul), ulong2(0x3fefd0001c77e49eul, 0x3c7181ee4bceacb1ul), ulong2(0x3fefeffff7e0c331ul, 0xbc6e05370170875aul), ulong2(0x3ff00ffff465606eul, 0xbc8a7ead491c0adaul), ulong2(0x3ff02ffff3867a58ul, 0xbc977f69c3fcb2e0ul), ulong2(0x3ff04ffffdfc0d17ul, 0x3c97bffe34cb945bul), ulong2(0x3ff0700003cd4d82ul, 0x3c820083c0e456cbul), ulong2(0x3ff08ffff9f2cbe8ul, 0xbc6dffdfbe37751aul), ulong2(0x3ff0b000010cda65ul, 0xbc913f7faee626ebul), ulong2(0x3ff0d00001a4d338ul, 0x3c807dfa79489ff7ul), ulong2(0x3ff0effffadafdfdul, 0xbc77040570d66bc0ul), ulong2(0x3ff110000bbafd96ul, 0x3c8e80d4846d0b62ul), ulong2(0x3ff12ffffae5f45dul, 0x3c9dbffa64fd36eful), ulong2(0x3ff150000dd59ad9ul, 0x3c9a0077701250aeul), ulong2(0x3ff170000f21559aul, 0x3c8dfdf9e2e3deeeul), ulong2(0x3ff18ffffc275426ul, 0x3c910030dc3b7273ul), ulong2(0x3ff1b000123d3c59ul, 0x3c997f7980030188ul), ulong2(0x3ff1cffff8299eb7ul, 0xbc65f932ab9f8c67ul), ulong2(0x3ff1effff48ad400ul, 0x3c937fbf9da75bebul), ulong2(0x3ff210000c8b86a4ul, 0x3c9f806b91fd5b22ul), ulong2(0x3ff2300003854303ul, 0x3c93ffc2eb9fbf33ul), ulong2(0x3ff24fffffbcf684ul, 0x3c7601e77e2e2e72ul), ulong2(0x3ff26ffff52921d9ul, 0x3c7ffcbb767f0c61ul), ulong2(0x3ff2900014933a3cul, 0xbc7202ca3c02412bul), ulong2(0x3ff2b00014556313ul, 0xbc92808233f21f02ul), ulong2(0x3ff2cfffebfe523bul, 0xbc88ff7e384fdcf2ul), ulong2(0x3ff2f0000bb8ad96ul, 0xbc85ff51503041c5ul), ulong2(0x3ff30ffffb7ae2aful, 0xbc810071885e289dul), ulong2(0x3ff32ffffeac5f7ful, 0xbc91ff5d3fb7b715ul), ulong2(0x3ff350000ca66756ul, 0x3c957f82228b82bdul), ulong2(0x3ff3700011fbf721ul, 0x3c8000bac40dd5ccul), ulong2(0x3ff38ffff9592fb9ul, 0xbc943f9d2db2a751ul), ulong2(0x3ff3b00004ddd242ul, 0x3c857f6b707638e1ul), ulong2(0x3ff3cffff5b2c957ul, 0x3c7a023a10bf1231ul), ulong2(0x3ff3efffeab0b418ul, 0x3c987f6d66b152b0ul), ulong2(0x3ff410001532aff4ul, 0x3c67f8375f198524ul), ulong2(0x3ff4300017478b29ul, 0x3c8301e672dc5143ul), ulong2(0x3ff44fffe795b463ul, 0x3c89ff69b8b2895aul), ulong2(0x3ff46fffe80475e0ul, 0xbc95c0b19bc2f254ul), ulong2(0x3ff48fffef6fc1e7ul, 0x3c9b4009f23a2a72ul), ulong2(0x3ff4afffe5bea704ul, 0xbc94ffb7bf0d7d45ul), ulong2(0x3ff4d000171027deul, 0xbc99c06471dc6a3dul), ulong2(0x3ff4f0000ff03ee2ul, 0x3c977f890b85531cul), ulong2(0x3ff5100012dc4bd1ul, 0x3c6004657166a436ul), ulong2(0x3ff530001605277aul, 0xbc96bfcece233209ul), ulong2(0x3ff54fffecdb704cul, 0xbc8902720505a1d7ul), ulong2(0x3ff56fffef5f54a9ul, 0x3c9bbfe60ec96412ul), ulong2(0x3ff5900017e61012ul, 0x3c887ec581afef90ul), ulong2(0x3ff5b00003c93e92ul, 0xbc9f41080abf0cc0ul), ulong2(0x3ff5d0001d4919bcul, 0xbc98812afb254729ul), ulong2(0x3ff5efffe7b87a89ul, 0xbc947eb780ed6904ul)};
#define TF_LN2HI_LOG 0x3fe62e42fefa3800ul
#define TF_LN2LO_LOG 0x3d2ef35793c76730ul

// binary64 arithmetic on bit patterns (ulong), integer operations only: round to nearest, ties to even; zeros,
// subnormals, infinities and NaNs as IEEE 754 (a NaN operand comes back quieted). Each operation gives the word the
// host's double operation gives.
#define TF_F64_SIGN 0x8000000000000000ul
#define TF_F64_FRAC 0x000FFFFFFFFFFFFFul
#define TF_F64_HIDDEN 0x0010000000000000ul
#define TF_F64_INF 0x7FF0000000000000ul
#define TF_F64_QNAN 0x7FF8000000000000ul

static inline uint tf_f64_exp(ulong x) { return uint(x >> 52) & 0x7FFu; }
static inline bool tf_f64_nan(ulong x) { return (x & ~TF_F64_SIGN) > TF_F64_INF; }
static inline bool tf_f64_zero(ulong x) { return (x << 1) == 0ul; }
// `m` shifted right by `n`, any bit shifted out ORed into bit 0
static inline ulong tf_jam(ulong m, uint n) {
  if (n == 0u) return m;
  if (n >= 64u) return m != 0ul ? 1ul : 0ul;
  return (m >> n) | ((m << (64u - n)) != 0ul ? 1ul : 0ul);
}
// the word of sign | m * 2^(e - 1085) rounded (m's leading bit at most bit 62; bits 9..0 below the result's last bit)
static inline ulong tf_round_pack(ulong sign, int e, ulong m) {
  if (e < 1) { m = tf_jam(m, uint(1 - e)); e = 1; }
  if (e >= 0x7FF) return sign | TF_F64_INF;
  const ulong low = m & 0x3FFul;
  ulong r = (m + 0x200ul) >> 10;
  if (low == 0x200ul) r &= ~1ul;
  ulong bits = (ulong(e - 1) << 52) + r;
  if (bits >= TF_F64_INF) bits = TF_F64_INF;
  return sign | bits;
}
static inline ulong tf_f64_add(ulong a, ulong b) {
  const uint ea = tf_f64_exp(a), eb = tf_f64_exp(b);
  if (ea == 0x7FFu || eb == 0x7FFu) {
    if (tf_f64_nan(a)) return a | 0x0008000000000000ul;
    if (tf_f64_nan(b)) return b | 0x0008000000000000ul;
    if (ea == 0x7FFu && eb == 0x7FFu && ((a ^ b) & TF_F64_SIGN) != 0ul) return TF_F64_QNAN;
    return ea == 0x7FFu ? a : b;
  }
  if (tf_f64_zero(a)) return tf_f64_zero(b) ? (a & b & TF_F64_SIGN) : b;
  if (tf_f64_zero(b)) return a;
  ulong ma = ((a & TF_F64_FRAC) | (ea != 0u ? TF_F64_HIDDEN : 0ul)) << 10;
  ulong mb = ((b & TF_F64_FRAC) | (eb != 0u ? TF_F64_HIDDEN : 0ul)) << 10;
  int xa = ea != 0u ? int(ea) : 1, xb = eb != 0u ? int(eb) : 1;
  ulong sa = a & TF_F64_SIGN, sb = b & TF_F64_SIGN;
  if (xa < xb || (xa == xb && ma < mb)) {
    const ulong tm = ma; ma = mb; mb = tm;
    const int tx = xa; xa = xb; xb = tx;
    const ulong ts = sa; sa = sb; sb = ts;
  }
  mb = tf_jam(mb, uint(xa - xb));
  int e = xa;
  ulong m;
  if (sa == sb) {
    m = ma + mb;
    if (m >= 0x8000000000000000ul) { m = (m >> 1) | (m & 1ul); e += 1; }
  } else {
    m = ma - mb;
    if (m == 0ul) return 0ul;
    const uint lz = clz(m) - 1u;
    m <<= lz;
    e -= int(lz);
  }
  return tf_round_pack(sa, e, m);
}
static inline ulong tf_f64_sub(ulong a, ulong b) { return tf_f64_add(a, b ^ TF_F64_SIGN); }
// a finite nonzero operand's significand with its leading bit at bit 52, and its exponent (subnormals normalized)
static inline ulong tf_f64_sig(ulong x, thread int& e) {
  const uint ex = tf_f64_exp(x);
  ulong f = x & TF_F64_FRAC;
  if (ex != 0u) { e = int(ex); return f | TF_F64_HIDDEN; }
  const uint shift = clz(f) - 11u;
  e = 1 - int(shift);
  return f << shift;
}
static inline ulong tf_f64_mul(ulong a, ulong b) {
  const ulong sign = (a ^ b) & TF_F64_SIGN;
  const uint ea = tf_f64_exp(a), eb = tf_f64_exp(b);
  if (tf_f64_nan(a)) return a | 0x0008000000000000ul;
  if (tf_f64_nan(b)) return b | 0x0008000000000000ul;
  if (ea == 0x7FFu || eb == 0x7FFu) return (tf_f64_zero(a) || tf_f64_zero(b)) ? TF_F64_QNAN : (sign | TF_F64_INF);
  if (tf_f64_zero(a) || tf_f64_zero(b)) return sign;
  int xa, xb;
  const ulong ma = tf_f64_sig(a, xa), mb = tf_f64_sig(b, xb);
  // the 106-bit product from 32-bit halves
  const ulong al = ma & 0xFFFFFFFFul, ah = ma >> 32, bl = mb & 0xFFFFFFFFul, bh = mb >> 32;
  const ulong ll = al * bl, mid = al * bh + ah * bl, hh = ah * bh;
  const ulong lo = ll + (mid << 32);
  const ulong hi = hh + (mid >> 32) + (lo < ll ? 1ul : 0ul);
  ulong m = (hi << 22) | (lo >> 42) | ((lo & 0x3FFFFFFFFFFul) != 0ul ? 1ul : 0ul);
  int e = xa + xb - 1023;
  if (m >= 0x8000000000000000ul) { m = (m >> 1) | (m & 1ul); e += 1; }
  return tf_round_pack(sign, e, m);
}
static inline ulong tf_f64_div(ulong a, ulong b) {
  const ulong sign = (a ^ b) & TF_F64_SIGN;
  const uint ea = tf_f64_exp(a), eb = tf_f64_exp(b);
  if (tf_f64_nan(a)) return a | 0x0008000000000000ul;
  if (tf_f64_nan(b)) return b | 0x0008000000000000ul;
  if (ea == 0x7FFu) return eb == 0x7FFu ? TF_F64_QNAN : (sign | TF_F64_INF);
  if (eb == 0x7FFu) return sign;
  if (tf_f64_zero(b)) return tf_f64_zero(a) ? TF_F64_QNAN : (sign | TF_F64_INF);
  if (tf_f64_zero(a)) return sign;
  int xa, xb;
  ulong r = tf_f64_sig(a, xa);
  const ulong mb = tf_f64_sig(b, xb);
  int e = xa - xb + 1023;
  if (r < mb) { r <<= 1; e -= 1; }
  ulong q = 0ul;
  for (uint i = 0u; i < 63u; ++i) {
    q <<= 1;
    if (r >= mb) { r -= mb; q |= 1ul; }
    r <<= 1;
  }
  return tf_round_pack(sign, e, q | (r != 0ul ? 1ul : 0ul));
}
// a < b, a > b, a == b for non-NaN words (-0 == +0)
static inline long tf_f64_key(ulong x) { return (x & TF_F64_SIGN) != 0ul ? -long(x & ~TF_F64_SIGN) : long(x); }
static inline bool tf_f64_lt(ulong a, ulong b) { return tf_f64_key(a) < tf_f64_key(b); }
static inline bool tf_f64_eq(ulong a, ulong b) { return tf_f64_key(a) == tf_f64_key(b); }
// @max of two non-NaN doubles as aarch64 fmaxnm gives it (+0 over -0)
static inline ulong tf_f64_max(ulong a, ulong b) {
  if (tf_f64_lt(a, b)) return b;
  if (tf_f64_lt(b, a)) return a;
  return a & b;
}
// an f32 word's double (exact)
static inline ulong tf_f64_of_f32(uint x) {
  const ulong sign = ulong(x & 0x80000000u) << 32;
  const uint e = (x >> 23) & 0xFFu, f = x & 0x7FFFFFu;
  if (e == 0xFFu) return sign | TF_F64_INF | (ulong(f) << 29) | (f != 0u ? 0x0008000000000000ul : 0ul);
  if (e == 0u) {
    if (f == 0u) return sign;
    const uint p = 31u - clz(f);  // leading bit: value f * 2^-149
    return sign | (ulong(p + 1023u - 149u) << 52) | ((ulong(f) << (52u - p)) & TF_F64_FRAC);
  }
  return sign | (ulong(e + 1023u - 127u) << 52) | (ulong(f) << 29);
}
// a bf16 word's double
static inline ulong tf_f64_of_bf16(ushort x) { return tf_f64_of_f32(uint(x) << 16); }
static inline ulong tf_f64_of_int(long v) {
  if (v == 0) return 0ul;
  const ulong sign = v < 0 ? TF_F64_SIGN : 0ul;
  const ulong m = v < 0 ? ulong(-v) : ulong(v);
  const uint p = 63u - clz(m);
  if (p <= 52u) return sign | (ulong(p + 1023u) << 52) | ((m << (52u - p)) & TF_F64_FRAC);
  return tf_round_pack(sign, int(p) + 1023, m << (62u - p));
}
// @intFromFloat to i32 (toward zero) of a finite double inside its range
static inline int tf_f64_trunc(ulong x) {
  const int e = int(tf_f64_exp(x)) - 1023;
  if (e < 0) return 0;
  const ulong m = (x & TF_F64_FRAC) | TF_F64_HIDDEN;
  const ulong v = e >= 52 ? (m << uint(e - 52)) : (m >> uint(52 - e));
  return (x & TF_F64_SIGN) != 0ul ? -int(v) : int(v);
}
// std.math.ldexp(f64), as written there
static inline ulong tf_f64_ldexp(ulong repr, int n) {
  if ((repr & ~TF_F64_SIGN) >= TF_F64_INF) return repr;
  const ulong sign_bit = repr & TF_F64_SIGN;
  int exponent = int((repr << 1) >> 53);
  if (exponent == 0) exponent += 11 - int(clz(repr << 1));
  if (n >= 0) {
    if (n > 2046 - exponent) return TF_F64_INF | sign_bit;
    if (exponent + n <= 0) return (repr << uint(n)) | sign_bit;
    if (exponent <= 0) {
      ulong result = ulong(n + exponent) << 52;
      result |= (repr << uint(1 - exponent)) & TF_F64_FRAC;
      return result | sign_bit;
    }
    return repr + (ulong(n) << 52);
  }
  if (n <= -exponent) {
    if (n < -(52 + exponent)) return sign_bit;
    const uint shift = uint(min(-n, -(exponent + n) + 1));
    const bool exact_tie = ctz(repr) == shift - 1u;
    ulong result = repr & TF_F64_FRAC;
    result |= ulong(exponent > 0 ? 1u : 0u) << 52;
    result = result >> (shift - 1u);
    result = ((result + 1ul) >> 1) & ~ulong(exact_tie ? 1u : 0u);
    return result | sign_bit;
  }
  return repr - (ulong(-n) << 52);
}
#define TF_ADD tf_f64_add
#define TF_SUB tf_f64_sub
#define TF_MUL tf_f64_mul
#define TF_DIV tf_f64_div
// compiler_rt exp_f64 (musl), operation by operation
static inline ulong tf_f64_expf(ulong x) {
  const ulong ln2hi = 0x3fe62e42fee00000ul, ln2lo = 0x3dea39ef35793c76ul, invln2 = 0x3ff71547652b82feul;
  const ulong P1 = 0x3fc555555555553eul, P2 = 0xbf66c16c16bebd93ul, P3 = 0x3f11566aaf25de2cul, P4 = 0xbebbbd41c5d26bf1ul, P5 = 0x3e66376972bea4d0ul;
  const ulong one = 0x3ff0000000000000ul, two = 0x4000000000000000ul;
  uint hx = uint(x >> 32);
  const uint sign = hx >> 31;
  hx &= 0x7FFFFFFFu;
  if (tf_f64_nan(x)) return x;
  if (hx >= 0x4086232Bu) {
    if (hx > 0x7FF00000u) return x;
    if (tf_f64_lt(0x40862e42fefa39eful, x)) return TF_F64_INF;
    if (tf_f64_lt(x, 0xc086232bdd7abcd2ul)) {
      if (tf_f64_lt(x, 0xc0874910d52d3051ul)) return 0ul;
    }
  }
  int k;
  ulong hi, lo;
  if (hx > 0x3FD62E42u) {
    if (hx > 0x3FF0A2B2u) k = tf_f64_trunc(TF_ADD(TF_MUL(invln2, x), sign != 0u ? 0xbfe0000000000000ul : 0x3fe0000000000000ul));
    else k = 1 - int(sign) - int(sign);
    const ulong dk = tf_f64_of_int(k);
    hi = TF_SUB(x, TF_MUL(dk, ln2hi));
    lo = TF_MUL(dk, ln2lo);
    x = TF_SUB(hi, lo);
  } else if (hx > 0x3E300000u) {
    k = 0;
    hi = x;
    lo = 0ul;
  } else {
    return TF_ADD(one, x);
  }
  const ulong xx = TF_MUL(x, x);
  const ulong c = TF_SUB(x, TF_MUL(xx, TF_ADD(P1, TF_MUL(xx, TF_ADD(P2, TF_MUL(xx, TF_ADD(P3, TF_MUL(xx, TF_ADD(P4, TF_MUL(xx, P5))))))))));
  const ulong y = TF_ADD(one, TF_ADD(TF_SUB(TF_DIV(TF_MUL(x, c), TF_SUB(two, c)), lo), hi));
  return k == 0 ? y : tf_f64_ldexp(y, k);
}
// compiler_rt log_f64 (musl), operation by operation
static inline ulong tf_f64_logf(ulong x) {
  long ix = long(x);
  if (0x3fee000000000000l <= ix && ix < 0x3ff1090000000000l) {
    if (x == 0x3ff0000000000000ul) return 0ul;
    const ulong r = TF_SUB(x, 0x3ff0000000000000ul);
    const ulong r2 = TF_MUL(r, r);
    const ulong r3 = TF_MUL(r, r2);
    #define P1N(i) tf_log_poly1[i]
    const ulong inner3 = TF_ADD(TF_ADD(TF_ADD(P1N(7), TF_MUL(r, P1N(8))), TF_MUL(r2, P1N(9))), TF_MUL(r3, P1N(10)));
    const ulong inner2 = TF_ADD(TF_ADD(TF_ADD(P1N(4), TF_MUL(r, P1N(5))), TF_MUL(r2, P1N(6))), TF_MUL(r3, inner3));
    const ulong inner1 = TF_ADD(TF_ADD(TF_ADD(P1N(1), TF_MUL(r, P1N(2))), TF_MUL(r2, P1N(3))), TF_MUL(r3, inner2));
    const ulong y = TF_MUL(r3, inner1);
    ulong w = TF_MUL(r, 0x41a0000000000000ul);
    const ulong rhi = TF_SUB(TF_ADD(r, w), w);
    const ulong rlo = TF_SUB(r, rhi);
    w = TF_MUL(TF_MUL(rhi, rhi), P1N(0));
    const ulong hi = TF_ADD(r, w);
    const ulong lo = TF_ADD(TF_ADD(TF_SUB(r, hi), w), TF_MUL(TF_MUL(P1N(0), rlo), TF_ADD(rhi, r)));
    #undef P1N
    return TF_ADD(TF_ADD(y, lo), hi);
  }
  const ulong top = ulong(ix) >> 48;
  if (top < 0x0010ul || 0x7ff0ul <= top) {
    if ((ulong(ix) << 1) == 0ul) return TF_F64_INF | TF_F64_SIGN;
    if (ulong(ix) == TF_F64_INF) return x;
    if ((top & 0x8000ul) != 0ul || (top & 0x7ff0ul) == 0x7ff0ul) return TF_F64_QNAN;
    ix = long(TF_MUL(x, 0x4330000000000000ul)) - (52l << 52);
  }
  const long tmp = ix - 0x3fe6000000000000l;
  const uint i = uint(tmp >> 45) & 0x7Fu;
  const long k = tmp >> 52;
  const ulong iz = ulong(ix) - (ulong(k) << 52);
  const ulong invc = tf_log_tab[i].x, logc = tf_log_tab[i].y;
  const ulong z = iz;
  const ulong r = TF_MUL(TF_SUB(TF_SUB(z, tf_log_tab2[i].x), tf_log_tab2[i].y), invc);
  const ulong kd = tf_f64_of_int(k);
  const ulong w = TF_ADD(TF_MUL(kd, TF_LN2HI_LOG), logc);
  const ulong hi = TF_ADD(w, r);
  const ulong lo = TF_ADD(TF_ADD(TF_SUB(w, hi), r), TF_MUL(kd, TF_LN2LO_LOG));
  const ulong r2 = TF_MUL(r, r);
  const ulong p = tf_log_poly[0];
  const ulong tail = TF_ADD(TF_ADD(tf_log_poly[1], TF_MUL(r, tf_log_poly[2])), TF_MUL(r2, TF_ADD(tf_log_poly[3], TF_MUL(r, tf_log_poly[4]))));
  const ulong y = TF_ADD(TF_ADD(TF_ADD(lo, TF_MUL(r2, p)), TF_MUL(TF_MUL(r, r2), tail)), hi);
  return y;
}

// The next round's draft tree of a depth-one block (one lattice row of `topk` candidates) as selector.propose builds it on
// the host: the anchor's predecessor row times the projected row, each candidate's successor row summed against it in
// rank order, the scores, their normalization, the drafter's floor probabilities, the siblings offered in score order
// (ties by index) and the tree in `before` order (ties by token).
//
// Two ways to the same tree. The exact way computes every double the host computes, operation by operation (binary64
// above), and makes the host's comparisons on them. The bounded way computes each candidate's score in double-f32 and its
// floor test in f32 with error bounds far below its margins: when every pair of scores differs by more than the score
// margin and every floor test clears its margin, the host's order and floor decisions are the ones these values give (each
// host rounding is monotone and its error lies inside the margins), so the tree is the host's. Otherwise, and on any input
// the bound does not cover, the exact way decides.
struct TfDraftSelect {
  uint vocab;
  uint topk;
  uint rank;
  uint nodes;
  uint children;
  uint mapped;
  uint map_width;
  uint both;  // 1: the exact way always runs too, its tree at word 0 and the bounded way's at word 256 (0xFFFFFFFF: deferred)
  ulong floor;
  ulong edge;
  ulong tau;
};

static inline float2 tf_two_sum(float a, float b) {
  const float s = a + b;
  const float bb = s - a;
  return float2(s, (a - (s - bb)) + (b - bb));
}
static inline float2 tf_quick_two_sum(float a, float b) {
  const float s = a + b;
  return float2(s, b - (s - a));
}
// (a.x + a.y) + (b.x + b.y) in double-f32
static inline float2 tf_df_add(float2 a, float2 b) {
  float2 s = tf_two_sum(a.x, b.x);
  const float2 t = tf_two_sum(a.y, b.y);
  s.y += t.x;
  s = tf_quick_two_sum(s.x, s.y);
  s.y += t.y;
  return tf_quick_two_sum(s.x, s.y);
}
static inline float tf_bf16(ushort w) { return as_type<float>(uint(w) << 16); }

// The exact way, for every thread of the group (its barriers are the group's): threads below 32 work.
static void tf_select_exact(device const uint* row, device const uint* map, device const ushort* projected, uint anchor,
                            device const ushort* pred, device const ushort* succ, device uint* out, constant TfDraftSelect& p,
                            uint lane, threadgroup ulong* terms, threadgroup ulong* unary, threadgroup ulong* score,
                            threadgroup ulong* d, threadgroup ulong* e, threadgroup ulong* q, threadgroup ulong* f,
                            threadgroup uint* cand, threadgroup uint* order, threadgroup ulong* shared_words,
                            threadgroup atomic_uint* flags) {
  uint bad = 0u;
  const ulong arow = ulong(anchor < p.vocab ? anchor : 0u) * 256ul;
  if (lane < 32u) {
    for (uint k = 0u; k < 8u; ++k) {
      const uint r = lane * 8u + k;
      terms[r] = tf_f64_mul(tf_f64_of_bf16(pred[arow + r]), tf_f64_of_bf16(projected[r]));
    }
  }
  if (lane < 16u) unary[lane] = tf_f64_of_f32(row[3u + lane * 2u]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 16u) {
    const ulong srow = ulong(cand[lane]) * 256ul;
    ulong sum = 0ul;
    for (uint r = 0u; r < 256u; ++r) sum = tf_f64_add(sum, tf_f64_mul(terms[r], tf_f64_of_bf16(succ[srow + r])));
    e[lane] = sum;
    ulong s = tf_f64_add(unary[lane], tf_f64_mul(p.edge, sum));
    s = tf_f64_div(s, p.tau);
    if (tf_f64_exp(s) == 0x7FFu) bad |= 16u;
    score[lane] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    ulong maximum = TF_F64_INF | TF_F64_SIGN, top = TF_F64_INF | TF_F64_SIGN;
    for (uint i = 0u; i < 16u; ++i) {
      maximum = tf_f64_max(maximum, score[i]);
      top = tf_f64_max(top, unary[i]);
    }
    shared_words[0] = maximum;
    shared_words[1] = top;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 16u) {
    d[lane] = tf_f64_sub(score[lane], shared_words[0]);
    f[lane] = tf_f64_expf(d[lane]);
    q[lane] = tf_f64_expf(tf_f64_sub(unary[lane], shared_words[1]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    ulong denominator = 0ul, total = 0ul;
    for (uint i = 0u; i < 16u; ++i) {
      denominator = tf_f64_add(denominator, f[i]);
      total = tf_f64_add(total, q[i]);
    }
    shared_words[2] = tf_f64_logf(denominator);
    shared_words[3] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 16u) {
    f[lane] = tf_f64_sub(d[lane], shared_words[2]);
    q[lane] = tf_f64_div(q[lane], shared_words[3]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 16u) {
    uint rank = 0u;
    for (uint j = 0u; j < 16u; ++j) rank += (tf_f64_lt(f[lane], f[j]) || (tf_f64_eq(f[lane], f[j]) && j < lane)) ? 1u : 0u;
    order[rank] = lane;
  }
  if (bad != 0u) atomic_fetch_or_explicit(flags, bad, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    uint taken[16];
    uint n = 0u;
    for (uint k = 0u; k < 16u; ++k) {
      const uint i = order[k];
      if (n == p.children) break;
      if (n > 0u && p.floor != 0ul && tf_f64_lt(q[i], p.floor)) continue;
      taken[n++] = i;
    }
    // before(): score descending, ties by token
    for (uint a = 1u; a < n; ++a) {
      const uint x = taken[a];
      uint b = a;
      while (b > 0u) {
        const uint y = taken[b - 1u];
        if (!(tf_f64_lt(f[y], f[x]) || (tf_f64_eq(f[y], f[x]) && cand[x] < cand[y]))) break;
        taken[b] = y;
        --b;
      }
      taken[b] = x;
    }
    n = min(n, p.nodes);
    out[0] = atomic_load_explicit(flags, memory_order_relaxed);
    out[1] = 1u + n;
    for (uint j = 0u; j < 8u; ++j) {
      out[2u + j] = j == 0u || j > n ? anchor : cand[taken[j - 1u]];
      out[10u + j] = j == 0u ? 0xFFFFFFFFu : (j <= n ? 0u : 0xFFFFFFFFu);
    }
  }
}

// out: [0] status (0: the tree below; else why the host's selector fails), [1] rows (the anchor and the drafted tokens),
// [2, 10) ids (the anchor first, and again in the rows past the tree's), [10, 18) parents (verify rows: -1, then 0 for each drafted token), [18] 1 when the exact
// way decided. One group of 512 threads: SIMD group c sums candidate c.
kernel void tf_draft_select(device const uint* row [[buffer(0)]], device const uint* map [[buffer(1)]],
                            device const ushort* projected [[buffer(2)]], device const uint* anchor_word [[buffer(3)]],
                            device const ushort* pred [[buffer(4)]], device const ushort* succ [[buffer(5)]],
                            device uint* out [[buffer(6)]], constant TfDraftSelect& p [[buffer(7)]],
                            uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float pterm[256];
  threadgroup float2 edge_df[16];
  threadgroup float edge_abs[16];
  threadgroup ulong terms[256];
  threadgroup ulong unary[16], score[16], d[16], e[16], q[16], f[16];
  threadgroup uint cand[16], order[16];
  threadgroup ulong shared_words[4];
  threadgroup atomic_uint flags;
  threadgroup atomic_uint slow;
  if (tid == 0u) {
    atomic_store_explicit(&flags, 0u, memory_order_relaxed);
    atomic_store_explicit(&slow, p.edge == 0x3fd3333333333333ul ? 0u : 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint bad = 0u, defer = 0u;
  const uint anchor = anchor_word[0];
  if (tid == 0u && (row[0] != 0u || row[1] != p.topk || p.topk != 16u || p.rank != 256u || p.nodes > 15u || p.children == 0u)) bad |= 1u;
  if (anchor >= p.vocab) bad |= 2u;
  const ulong arow = ulong(anchor < p.vocab ? anchor : 0u) * 256ul;
  // the anchor's predecessor row times the projected row (16-bit significands: exact in f32 in the normal range)
  if (tid < 256u) {
    const ushort pw = pred[arow + tid], xw = projected[tid];
    if ((pw & 0x7F80u) == 0x7F80u) bad |= 4u;
    if ((xw & 0x7F80u) == 0x7F80u) bad |= 8u;
    const float a = tf_bf16(pw), b = tf_bf16(xw);
    if (fabs(a) >= 0x1p30f || fabs(b) >= 0x1p30f) defer = 1u;
    pterm[tid] = a * b;
  }
  if (tid < 16u) {
    const uint token = row[2u + tid * 2u], word = row[3u + tid * 2u];
    uint c = token;
    if (p.mapped != 0u) c = token < p.map_width ? map[token] : 0xFFFFFFFFu;
    if (token == 0xFFFFFFFFu || c >= p.vocab) { bad |= 2u; c = 0u; }
    if ((word & 0x7F800000u) == 0x7F800000u || (word & 0xFFFFu) != 0u) bad |= 1u;
    cand[tid] = c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // candidate sg's edge: its successor row against the terms, eight rank positions a lane, in double-f32
  {
    const ulong srow = ulong(cand[sg & 15u]) * 256ul;
    float2 acc = float2(0.0f);
    float mag = 0.0f;
    for (uint k = 0u; k < 8u; ++k) {
      const uint r = lane * 8u + k;
      const ushort sw = succ[srow + r];
      if ((sw & 0x7F80u) == 0x7F80u) bad |= 4u;
      const float s = tf_bf16(sw);
      if (fabs(s) >= 0x1p30f) defer = 1u;
      const float t = pterm[r] * s;
      acc = tf_df_add(acc, float2(t, 0.0f));
      mag += fabs(t);
    }
    for (ushort off = 16; off > 0; off >>= 1) {
      const float2 other = float2(simd_shuffle_down(acc.x, off), simd_shuffle_down(acc.y, off));
      acc = tf_df_add(acc, other);
      mag += simd_shuffle_down(mag, off);
    }
    if (lane == 0u && sg < 16u) {
      edge_df[sg] = acc;
      edge_abs[sg] = mag;
    }
  }
  if (bad != 0u) atomic_fetch_or_explicit(&flags, bad, memory_order_relaxed);
  if (defer != 0u) atomic_store_explicit(&slow, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the bounded decisions, on SIMD group 0's first 16 lanes
  if (sg == 0u) {
    const uint c = lane & 15u;
    const float u = as_type<float>(row[3u + c * 2u]);
    // v = u + 0.3 edge (the score times 1.5: the same order)
    const float2 ed = edge_df[c];
    const float ch = as_type<float>(0x3E99999Au), cl = as_type<float>(0xB24CCCCDu);
    const float ph = ch * ed.x;
    const float pl = fma(ch, ed.x, -ph) + (ch * ed.y + cl * ed.x);
    float2 v = tf_two_sum(u, ph);
    v.y += pl;
    v = tf_quick_two_sum(v.x, v.y);
    const float mag = fabs(u) + 0.3f * edge_abs[c];
    const float scale = simd_max(mag) + 8.0f;
    const float margin = scale * 0x1p-30f + 0x1p-80f;
    const float top = simd_max(u);
    const float x = u - top;
    const float ex = x < -64.0f ? 0.0f : precise::exp(x);
    float total = ex;
    for (ushort off = 8; off > 0; off >>= 1) total += simd_shuffle_xor(total, off);
    const float floor32 = as_type<float>(0x3CA3D70Au);  // 0.02f
    const float bar = floor32 * total;
    const bool below = ex < bar;
    uint unsure = (fabs(ex - bar) <= 0x1p-16f * max(ex, bar) + 0x1p-100f) ? 1u : 0u;
    if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(total) || !isfinite(margin)) unsure = 1u;
    // rank by v descending (ties by index); any pair inside the margin defers
    uint rank = 0u;
    for (ushort j = 0; j < 16; ++j) {
      const float vjx = simd_shuffle(v.x, j), vjy = simd_shuffle(v.y, j);
      const float diff = (vjx - v.x) + (vjy - v.y);
      if (j != c && fabs(diff) <= margin) unsure = 1u;
      rank += (diff > 0.0f || (diff == 0.0f && j < c)) ? 1u : 0u;
    }
    if (lane < 16u) order[rank] = c;
    const uint any_unsure = simd_max(lane < 16u ? unsure : 0u);
    const uint below_bits = simd_or(lane < 16u && below ? (1u << c) : 0u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
      if (any_unsure != 0u || p.floor != 0x3f947ae147ae147bul) atomic_store_explicit(&slow, 1u, memory_order_relaxed);
      uint taken[16];
      uint n = 0u;
      for (uint k = 0u; k < 16u; ++k) {
        const uint i = order[k];
        if (n == p.children) break;
        if (n > 0u && ((below_bits >> i) & 1u) != 0u) continue;
        taken[n++] = i;
      }
      n = min(n, p.nodes);
      const uint base = p.both != 0u ? 256u : 0u;
      const bool decided = atomic_load_explicit(&slow, memory_order_relaxed) == 0u && atomic_load_explicit(&flags, memory_order_relaxed) == 0u;
      out[base] = decided ? 0u : 0xFFFFFFFFu;
      out[base + 1u] = 1u + n;
      for (uint j = 0u; j < 8u; ++j) {
        out[base + 2u + j] = j == 0u || j > n ? anchor : cand[taken[j - 1u]];
        out[base + 10u + j] = j == 0u ? 0xFFFFFFFFu : (j <= n ? 0u : 0xFFFFFFFFu);
      }
      out[base + 18u] = 0u;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (p.both != 0u || atomic_load_explicit(&slow, memory_order_relaxed) != 0u || atomic_load_explicit(&flags, memory_order_relaxed) != 0u) {
    tf_select_exact(row, map, projected, anchor, pred, succ, out, p, tid, terms, unary, score, d, e, q, f, cand, order, shared_words, &flags);
    if (tid == 0u) out[18] = 1u;
  }
}
