host := gpu2
PROJ := $(notdir ${CURDIR})

LOCAL := ${CURDIR}/src/
REMOTE := $(host):~/workspace/defense/

put : dbg
	rsync -abiLvzh --backup-dir='.rsync.bak' --delete-after \
		-e 'ssh -q' \
		--filter 'merge rsync-put.txt' \
		$(LOCAL) $(REMOTE)

get : dbg
	rsync -abiLvzh --backup-dir='.rsync.bak' --delete-after \
		-e 'ssh -q' \
		--filter 'merge rsync-get.txt' \
		$(REMOTE) $(LOCAL)

dbg :
	$(info host is $(host))

clean :
	rm *~

.PHONY : clean put get
